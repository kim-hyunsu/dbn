from tqdm import tqdm
import argparse
from flax import jax_utils
from giung2.data.build import build_dataloaders
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from giung2.models.layers import FilterResponseNorm
from models.resnet import FlaxResNet
import sghmc_deprecated
import sghmc
from functools import partial
import flax
from easydict import EasyDict
import defaults_sghmc as defaults
import os
import numpy as np
from utils import get_info_in_dir, normalize, model_list, logit_dir_list, feature_dir_list, jprint


def get_ckpt_temp(ckpt_dir):
    if not ckpt_dir.endswith("sghmc"):
        sghmc_ckpt_dir = os.path.join(ckpt_dir, "sghmc")
    sghmc_ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=sghmc_ckpt_dir, target=None)
    sghmc_config = EasyDict(sghmc_ckpt["config"])
    dataloaders = build_dataloaders(sghmc_config)
    del sghmc_ckpt["model"]

    # specify precision
    model_dtype = jnp.float32
    if sghmc_config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build model
    _ResNet = partial(
        FlaxResNet,
        depth=sghmc_config.model_depth,
        widen_factor=sghmc_config.model_width,
        dtype=model_dtype,
        pixel_mean=defaults.PIXEL_MEAN,
        pixel_std=defaults.PIXEL_STD,
        num_classes=dataloaders['num_classes'])

    if sghmc_config.model_style == 'BN-ReLU':
        model = _ResNet()
    elif sghmc_config.model_style == "FRN-Swish":
        model = _ResNet(
            conv=partial(
                flax.linen.Conv,
                use_bias=True,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)
    else:
        raise Exception("Unknown model_style")

    rng = jax.random.PRNGKey(sghmc_config.seed)
    init_rng, rng = jax.random.split(rng)

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones(dataloaders['image_shape'], model_dtype))
    variables = initialize_model(init_rng, model)

    if variables.get("batch_stats") is not None:
        sghmc_state = sghmc.get_sghmc_state(
            sghmc_config, dataloaders, model, variables)
    else:
        sghmc_state = sghmc_deprecated.get_sghmc_state(
            sghmc_config, dataloaders, model, variables)

    sghmc_ckpt["model"] = sghmc_state
    # sghmc_ckpt["ckpt_dir"] = sghmc_ckpt_dir
    return sghmc_config, sghmc_ckpt, rng


def load_classifer_state(model_dir, sample_idx, template):
    sghmc_ckpt_dir = model_dir
    if not sghmc_ckpt_dir.endswith("sghmc"):
        sghmc_ckpt_dir = os.path.join(sghmc_ckpt_dir, "sghmc")
    sghmc_ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=sghmc_ckpt_dir, target=template, step=sample_idx
    )
    state = sghmc_ckpt["model"]
    return state, sghmc_ckpt


class Bank():
    def __init__(self, num_classes, maxlen=128):
        self.bank = [jnp.array([]) for _ in range(num_classes)]
        self.len = [0 for _ in range(num_classes)]
        self.num_classes = num_classes
        self.maxlen = maxlen
        self.cached = None

    def _squeeze(self, batch):
        assert len(batch["images"].shape) == 5
        images = batch["images"]
        images = images.reshape(-1, *images.shape[2:])
        labels = batch["labels"].reshape(-1, 1)
        marker = batch["marker"].reshape(-1, 1)
        return images, labels, marker

    def _unpack(self, batch):
        assert len(batch["images"].shape) == 4
        images = batch["images"]
        labels = batch["labels"]
        marker = batch["marker"]
        return images, labels, marker

    def deposit(self, batch):
        images, labels, marker = self._unpack(batch)
        self._deposit(images, labels, marker)

    def _deposit(self, images, labels, marker):
        shape = images.shape
        images = images[marker, ...]
        if self.cached is None:
            self.bank = list(
                map(lambda x: x.reshape(-1, *shape[1:]), self.bank))

        # def func(i, val):
        def func(i):
            in_class = images[labels == i, ...]
            length = len(in_class)
            exceed = len(self.bank[i]) + length - self.maxlen
            if exceed > 0:
                self.bank[i] = self.bank[i][exceed:]
            self.bank[i] = jnp.concatenate([self.bank[i], in_class], axis=0)
            self.len[i] = len(self.bank[i])
            return val
        # val = jax.lax.fori_loop(0, self.num_classes, func, None)
        val = map(func, range(self.num_classes))
        min_len = min(self.len)
        # self.cached = jnp.array([t[-min_len:] for t in self.bank])

        def func(x):
            return x[-min_len:]
        cached = list(map(func, self.bank))
        self.cached = jnp.stack(cached)

    def withdraw(self, rng, batch):
        images, labels, marker = self._unpack(batch)
        assert len(images.shape) == 4
        out = self._withdraw(rng, labels)
        if out is None:
            return images
        assert out.shape == images.shape
        return out

    def _withdraw(self, rng, labels):
        min_len = min(self.len)
        if min_len == 0:
            return None
        indices = jax.random.randint(rng, (len(labels),), 0, min_len)
        new = self.cached[labels, indices]
        return new


def mixup_data(rng, batch, alpha=1.0):
    x = batch["images"]
    assert len(x.shape) == 4  # no parallel dimension

    beta_rng, perm_rng = jax.random.split(rng)
    lam = jnp.where(alpha > 0, jax.random.beta(beta_rng, alpha, alpha), 1)

    arange = jnp.arange(0, x.shape[0])
    index = jax.random.permutation(perm_rng, arange)
    index = jnp.where(x.shape[0] == jnp.sum(batch["marker"]), index, arange)

    mixed_x = lam*x+(1-lam)*x[index, ...]
    batch["images"] = mixed_x
    batch["lambdas"] = jnp.tile(lam, reps=[x.shape[0]])
    return batch


def mixup_inclass(rng, batch, bank, alpha=1.0):
    rng = jax_utils.unreplicate(rng)
    shapes = dict(
        images=batch["images"].shape,
        labels=batch["labels"].shape,
        marker=batch["marker"].shape)
    assert len(shapes["images"]) == 5
    batch["images"] = batch["images"].reshape(-1, *shapes["images"][2:])
    batch["labels"] = batch["labels"].reshape(-1, *shapes["labels"][2:])
    batch["marker"] = batch["marker"].reshape(-1, *shapes["marker"][2:])

    x = batch["images"]
    y = batch["labels"]
    bank.deposit(batch)

    beta_rng, perm_rng = jax.random.split(rng)
    lam = jnp.where(alpha > 0, jax.random.beta(beta_rng, alpha, alpha), 1)
    ingredient = bank.withdraw(perm_rng, batch)

    mixed_x = lam*x + (1-lam)*ingredient
    batch["images"] = mixed_x.reshape(*shapes["images"])
    batch["lambdas"] = jnp.tile(
        lam, reps=[x.shape[0]]).reshape(*shapes["labels"])
    batch["labels"] = batch["labels"].reshape(*shapes["labels"])
    batch["marker"] = batch["marker"].reshape(*shapes["marker"])
    return batch


if __name__ == "__main__":
    """
    'logits' and 'features' are used compatibly as terms of any features of a given data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="features_1mixupplus10", type=str)
    args = parser.parse_args()
    dir = args.dir

    settings = __import__(f"{dir}.settings", fromlist=[""])
    data_name = settings.data_name
    model_style = settings.model_style
    config, ckpt, rng = get_ckpt_temp(model_list(data_name, model_style)[0])
    config.optim_bs = 512
    n_samples_each_mode = 30
    n_modes = len(model_list(data_name, model_style))
    dataloaders = build_dataloaders(config)
    alpha, repeats = get_info_in_dir(dir)

    def get_logits(state, batch, feature_name="feature.vector"):
        x = batch["images"]
        marker = batch["marker"]
        params_dict = dict(params=state.params)
        mutable = ["intermediates"]
        if state.image_stats is not None:
            params_dict["image_stats"] = state.image_stats
        if hasattr(state, "batch_stats") and state.batch_stats is not None:
            params_dict["batch_stats"] = state.batch_stats
            mutable.append("batch_stats")
        _, new_model_state = state.apply_fn(
            params_dict, x, rngs=None, mutable=mutable)
        logits = new_model_state["intermediates"][feature_name][0]
        logits = jnp.where(marker[..., None], logits, jnp.zeros_like(logits))
        return logits

    if dir in logit_dir_list:
        p_get_logits = jax.pmap(
            partial(get_logits, feature_name="cls.logit"), axis_name="batch")
    elif dir in feature_dir_list:
        p_get_logits = jax.pmap(get_logits, axis_name="batch")
    else:
        raise Exception("Invalid directory for saving features")

    if data_name == "CIFAR100_x32":
        num_classes = 100
    elif data_name == "CIFAR10_x32":
        num_classes = 10

    bank = Bank(num_classes)
    if "mixupplus" in dir:
        p_mixup_data = partial(mixup_inclass, bank=bank, alpha=alpha)
    else:
        p_mixup_data = jax.pmap(
            partial(mixup_data, alpha=alpha), axis_name="batch")

    data_rng, rng = jax.random.split(rng)
    for mode_idx in range(n_modes):
        for i in tqdm(range(n_samples_each_mode)):
            feature_path = f"{dir}/train_features_M{mode_idx}S{i}.npy"
            lambda_path = f"{dir}/train_lambdas_M{mode_idx}S{i}.npy"
            if os.path.exists(feature_path):
                continue

            model_dir = model_list(data_name, model_style)[mode_idx]
            classifier_state, _ = load_classifer_state(model_dir, i+1, ckpt)
            classifier_state = jax_utils.replicate(classifier_state)
            # train set
            logits_list = []
            lambdas_list = []
            for rep in range(repeats):
                train_rng = jax.random.fold_in(rng, rep)
                _, mixup_rng = jax.random.split(train_rng)
                train_loader = dataloaders["trn_loader"](rng=None)
                train_loader = jax_utils.prefetch_to_device(
                    train_loader, size=2)
                for batch_idx, batch in enumerate(train_loader, start=1):
                    b_mixup_rng = jax.random.fold_in(mixup_rng, batch_idx)
                    batch = p_mixup_data(
                        jax_utils.replicate(b_mixup_rng), batch)
                    _logits = p_get_logits(classifier_state, batch)
                    logits = _logits[batch["marker"] ==
                                     True].reshape(-1, _logits.shape[-1])
                    assert len(logits.shape) == 2
                    lambdas = batch["lambdas"][batch["marker"] == True]
                    assert len(lambdas.shape) == 1
                    logits_list.append(logits)
                    lambdas_list.append(lambdas)
            logits = jnp.concatenate(logits_list, axis=0)
            lambdas = jnp.concatenate(lambdas_list, axis=0)
            with open(feature_path, "wb") as f:
                logits = np.array(logits)
                np.save(f, logits)
            with open(lambda_path, "wb") as f:
                lambdas = np.array(lambdas)
                np.save(f, lambdas)
            del logits_list
            del lambdas_list

            # valid set
            valid_loader = dataloaders["val_loader"](rng=None)
            valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
            logits_list = []
            # lambdas_list = []
            for batch_idx, batch in enumerate(valid_loader, start=1):
                _logits = p_get_logits(classifier_state, batch)
                logits = _logits[batch["marker"] ==
                                 True].reshape(-1, _logits.shape[-1])
                assert len(logits.shape) == 2
                # lambdas = batch["lambdas"][batch["marker"] == True]
                # assert len(lambdas.shape) == 1
                logits_list.append(logits)
                # lambdas_list.append(lambdas)
            logits = jnp.concatenate(logits_list, axis=0)
            # lambdas = jnp.concatenate(lambdas_list, axis=0)
            with open(f"{dir}/valid_features_M{mode_idx}S{i}.npy", "wb") as f:
                logits = np.array(logits)
                np.save(f, logits)
            # with open(f"{dir}/valid_lambdas_M{mode_idx}S{i}.npy", "wb") as f:
            #     lambdas = np.array(lambdas)
            #     np.save(f, lambdas)
            del logits_list
            # del lambdas_list

            # test set
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            logits_list = []
            # lambdas_list = []
            for batch_idx, batch in enumerate(test_loader, start=1):
                _logits = p_get_logits(classifier_state, batch)
                logits = _logits[batch["marker"] ==
                                 True].reshape(-1, _logits.shape[-1])
                assert len(logits.shape) == 2
                # lambdas = batch["lambdas"][batch["marker"] == True]
                # assert len(lambdas.shape) == 1
                logits_list.append(logits)
                # lambdas_list.append(lambdas)
            logits = jnp.concatenate(logits_list, axis=0)
            # lambdas = jnp.concatenate(lambdas_list, axis=0)
            with open(f"{dir}/test_features_M{mode_idx}S{i}.npy", "wb") as f:
                logits = np.array(logits)
                np.save(f, logits)
            # with open(f"{dir}/test_lambdas_M{mode_idx}S{i}.npy", "wb") as f:
            #     lambdas = np.array(lambdas)
            #     np.save(f, lambdas)
            del logits_list
            # del lambdas_list
