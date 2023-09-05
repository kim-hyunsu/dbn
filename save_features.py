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
from utils import expand_to_broadcast, get_info_in_dir, normalize, model_list, jprint
from utils import logit_dir_list, feature_dir_list, feature2_dir_list, feature3_dir_list, layer2stride1_dir_list
from flax.training import common_utils
import sgd_trainstate
from flax.core.frozen_dict import freeze


def get_ckpt_temp(ckpt_dir, shared_head=False, sgd=False):
    if not sgd:
        sghmc_ckpt_dir = os.path.join(ckpt_dir, "sghmc")
    else:
        sghmc_ckpt_dir = ckpt_dir
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

    if not sgd:
        if variables.get("batch_stats") is not None:
            if shared_head:
                sghmc_state = sghmc.get_sghmc_state(
                    sghmc_config, dataloaders, model, variables)
            else:
                sghmc_state = sghmc.get_sghmc_state_legacy(
                    sghmc_config, dataloaders, model, variables)
        else:
            sghmc_state = sghmc_deprecated.get_sghmc_state(
                sghmc_config, dataloaders, model, variables)
    else:
        sghmc_state = sgd_trainstate.get_sgd_state(
            sghmc_config, dataloaders, model, variables)

    sghmc_ckpt["model"] = sghmc_state
    return sghmc_config, sghmc_ckpt, rng


def load_classifier_state(model_dir, sample_idx, template, sgd=False, prev_state=None):
    sghmc_ckpt_dir = model_dir
    if prev_state is not None:
        if sample_idx > 1:
            return None, None
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=sghmc_ckpt_dir, target=None
        )
        prev_state = prev_state.replace(params=freeze(ckpt["model"]["params"]))
        return prev_state, ckpt
    if not sgd:
        sghmc_ckpt_dir = os.path.join(sghmc_ckpt_dir, "sghmc")
        sghmc_ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=sghmc_ckpt_dir, target=template, step=sample_idx
        )
    else:
        if sample_idx > 1:
            return None, None
        else:
            sghmc_ckpt = checkpoints.restore_checkpoint(
                ckpt_dir=sghmc_ckpt_dir, target=template
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


def mixup_data(rng, batch, alpha=1.0, interpolation=True):
    x = batch["images"]
    assert len(x.shape) == 4  # no parallel dimension

    beta_rng, perm_rng = jax.random.split(rng)
    lam = jnp.where(alpha > 0, jax.random.beta(beta_rng, alpha, alpha), 1)

    def mix():
        arange = jnp.arange(0, x.shape[0])
        index = jax.random.permutation(perm_rng, arange)
        index = jnp.where(x.shape[0] == jnp.sum(
            batch["marker"]), index, arange)

        mixed_x = jnp.where(
            interpolation,
            lam*x+(1-lam)*x[index, ...],
            (2-lam)*x-(1-lam)*x[index, ...])
        return mixed_x

    batch["images"] = jnp.where(
        alpha <= 0,
        batch["images"],
        mix()
    )
    batch["lambdas"] = jnp.tile(lam, reps=[x.shape[0]])
    return batch


def mixup_inclass(rng, batch, bank, alpha=1.0, interpolation=True):
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

    mixed_x = jnp.where(
        interpolation,
        lam*x + (1-lam)*ingredient,
        (2-lam)*x-(1-lam)*ingredient)
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
    shared_head = settings.shared_head
    sgd_state = getattr(settings, "sgd_state", False)
    bezier = "bezier" in dir
    distill = "distill" in dir
    distref = "distref" in dir
    distA = "distA" in dir
    distB = "distB" in dir
    AtoB = "AtoB" in dir
    AtoshB = "AtoshB" in dir
    AtoshABC = "AtoshABC" in dir
    AtoABC = "AtoABC" in dir
    layer2stride1_shared = "layer2stride1_shared" in dir
    tag = ""
    if bezier:
        tag = "bezier"
    elif distill:
        tag = "distill"
    elif distref:
        tag = "distref"
    elif distA:
        tag = "distA"
    elif distB:
        tag = "distB"
    elif AtoB:
        tag = "AtoB"
    elif AtoshB:
        tag = "AtoshB"
    elif AtoshABC:
        tag = "AtoshABC"
    elif AtoABC:
        tag = "AtoABC"
    elif layer2stride1_shared:
        tag = "layer2stride1_shared"
    model_dir_list = model_list(
        data_name, model_style, shared_head, tag)
    config, ckpt, rng = get_ckpt_temp(
        model_dir_list[0], shared_head, sgd_state)
    config.optim_bs = 512
    n_samples_each_mode = 1
    n_modes = len(model_dir_list)
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
        _marker = expand_to_broadcast(marker, logits, axis=1)
        logits = jnp.where(_marker, logits, jnp.zeros_like(logits))
        return logits

    if dir in logit_dir_list:
        p_get_logits = jax.pmap(
            partial(get_logits, feature_name="cls.logit"), axis_name="batch")
    elif dir in feature_dir_list:
        p_get_logits = jax.pmap(get_logits, axis_name="batch")
    elif dir in feature2_dir_list:
        p_get_logits = jax.pmap(
            partial(get_logits, feature_name="feature.layer3"))
    elif dir in feature3_dir_list:
        p_get_logits = jax.pmap(
            partial(get_logits, feature_name="feature.layer3stride2"))
    elif dir in layer2stride1_dir_list:
        p_get_logits = jax.pmap(
            partial(get_logits, feature_name="feature.layer2stride1"))
    else:
        raise Exception("Invalid directory for saving features")

    if data_name == "CIFAR100_x32":
        num_classes = 100
    elif data_name == "CIFAR10_x32":
        num_classes = 10

    bank = Bank(num_classes)
    if "ext" in dir:
        mixup_data = partial(mixup_data, interpolation=False)
        mixup_inclass = partial(mixup_inclass, interpolation=False)

    if "mixupplus" in dir:
        p_mixup_data = partial(mixup_inclass, bank=bank, alpha=alpha)
    else:
        if "rand" in dir:
            p_mixup_data = jax.pmap(
                mixup_data, axis_name="batch")
        else:
            p_mixup_data = jax.pmap(
                partial(mixup_data, alpha=alpha), axis_name="batch")

    @partial(jax.pmap, axis_name="batch")
    def adv_attack(state, batch, rng):
        def pred(x):
            params_dict = dict(params=state.params)
            mutable = ["intermediates"]
            if state.image_stats is not None:
                params_dict["image_stats"] = state.image_stats
            if hasattr(state, "batch_stats") and state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
                mutable.append("batch_stats")
            _, new_model_state = state.apply_fn(
                params_dict, x, rngs=None, mutable=mutable)
            logits = new_model_state['intermediates']['cls.logit'][0]
            return logits

        def loss_fn(x, labels):
            logits = pred(x)
            target = common_utils.onehot(
                labels, num_classes=logits.shape[-1])  # [B, K,]
            loss = -jnp.sum(target * jax.nn.log_softmax(logits,
                            axis=-1), axis=-1)      # [B,]
            loss = jnp.sum(
                jnp.where(batch['marker'], loss, jnp.zeros_like(loss))
            ) / jnp.sum(batch['marker'])
            return loss

        def ods_fn(x, w):
            p = jax.nn.softmax(pred(x))
            return jnp.sum(w*p)
        x = batch["images"]
        ####### version1 ##########
        # labels = batch["labels"]
        ####### version2 ##########
        # labels = jnp.argmax(pred(x))
        ###########################
        # Sign of gradient of the loss function
        ######## version1,2 #######
        # eps = jnp.sign(jax.grad(loss_fn)(x, labels))
        ######## version3 #########
        B = x.shape[0]
        d = num_classes
        w = jax.random.bernoulli(rng, 0.5, (B, d))
        eps = jax.grad(ods_fn)(x, w)
        eps = eps / \
            jnp.sqrt(jnp.sum(eps**2, axis=[1, 2, 3], keepdims=True))
        ######## version4 #########
        # eps = jax.random.normal(rng, x.shape) * 255 / (32*32*3)
        ############################
        adv_lmda = 1. / 255  # level of adversarial attack
        x_adv = x + eps * adv_lmda
        batch["images"] = x_adv
        return batch

    shape = None
    data_rng, rng = jax.random.split(rng)
    # temp = ckpt["model"]
    saved_classifier_state = None
    for mode_idx in range(n_modes):

        model_dir = model_dir_list[mode_idx]

        # if mode_idx != 0:
        _, ckpt, _ = get_ckpt_temp(
            model_dir, shared_head, sgd_state)

        if "bezier" in model_dir:
            # change here if bezier seed is changed
            assert "sd25" in model_dir and "bezier25" in dir
            ckpt["model"] = sgd_trainstate.TrainState2TrainStateRNG(
                ckpt["model"])
        # else:
        #     ckpt["model"] = temp

        for i in tqdm(range(n_samples_each_mode)):
            _classifier_state, _ = load_classifier_state(
                model_dir, i+1, ckpt, sgd_state, prev_state=saved_classifier_state)
            if _classifier_state is None:
                continue
            else:
                saved_classifier_state = _classifier_state
            classifier_state = jax_utils.replicate(_classifier_state)
            # train set
            feature_path = f"{dir}/train_features_M{mode_idx}S{i}.npy"
            if not os.path.exists(feature_path):
                lambda_path = f"{dir}/train_lambdas_M{mode_idx}S{i}.npy"
                logits_list = []
                lambdas_list = []
                for rep in range(repeats):
                    train_rng = jax.random.fold_in(rng, rep)
                    _, mixup_rng = jax.random.split(train_rng)
                    if "valid" in dir:
                        train_loader = dataloaders["val_loader"](rng=None)
                    else:
                        train_loader = dataloaders["trn_loader"](rng=None)
                    train_loader = jax_utils.prefetch_to_device(
                        train_loader, size=2)
                    for batch_idx, batch in enumerate(train_loader, start=1):
                        b_mixup_rng = jax.random.fold_in(mixup_rng, batch_idx)
                        if rep == 0:
                            ps, bs = batch["marker"].shape
                            batch["lambdas"] = jnp.tile(1, reps=[ps, bs])
                        else:
                            if "rand" in dir:
                                _alpha = jax.random.uniform(
                                    b_mixup_rng, (), minval=0, maxval=alpha)
                                _alpha = jax_utils.replicate(_alpha)
                                batch = p_mixup_data(
                                    jax_utils.replicate(b_mixup_rng), batch, _alpha)
                            else:
                                batch = p_mixup_data(
                                    jax_utils.replicate(b_mixup_rng), batch)
                        _logits = p_get_logits(classifier_state, batch)
                        logits = _logits[batch["marker"] == True]
                        logits = logits.reshape(-1, *_logits.shape[2:])
                        assert len(logits.shape) == 2 or len(logits.shape) == 4
                        lambdas = batch["lambdas"][batch["marker"] == True]
                        assert len(lambdas.shape) == 1
                        logits_list.append(logits)
                        lambdas_list.append(lambdas)
                logits = jnp.concatenate(logits_list, axis=0)
                shape = shape or logits.shape[1:]
                if shape:
                    assert logits.shape[1:] == shape
                lambdas = jnp.concatenate(lambdas_list, axis=0)
                with open(feature_path, "wb") as f:
                    logits = np.array(logits)
                    np.save(f, logits)
                with open(lambda_path, "wb") as f:
                    lambdas = np.array(lambdas)
                    np.save(f, lambdas)
                del logits_list
                del lambdas_list
            else:
                advfeature_path = f"{dir}/train_advfeatures_M{mode_idx}S{i}.npy"
                logits_list = []
                for rep in range(repeats):
                    # rngs
                    train_rng = jax.random.fold_in(rng, rep)
                    _, mixup_rng = jax.random.split(train_rng)
                    # choose type of loader
                    if "valid" in dir:
                        train_loader = dataloaders["val_loader"](rng=None)
                    else:
                        train_loader = dataloaders["trn_loader"](rng=None)
                    # prefetching
                    train_loader = jax_utils.prefetch_to_device(
                        train_loader, size=2)
                    # sample images
                    for batch_idx, batch in enumerate(train_loader, start=1):
                        b_mixup_rng = jax.random.fold_in(mixup_rng, batch_idx)
                        # mixup
                        if rep == 0:
                            ps, bs = batch["marker"].shape
                            batch["lambdas"] = jnp.tile(1, reps=[ps, bs])
                        else:
                            if "rand" in dir:
                                _alpha = jax.random.uniform(
                                    b_mixup_rng, (), minval=0, maxval=alpha)
                                _alpha = jax_utils.replicate(_alpha)
                                batch = p_mixup_data(
                                    jax_utils.replicate(b_mixup_rng), batch, _alpha)
                            else:
                                batch = p_mixup_data(
                                    jax_utils.replicate(b_mixup_rng), batch)
                        batch = adv_attack(
                            classifier_state, batch, jax_utils.replicate(b_mixup_rng))
                        # get logits
                        _logits = p_get_logits(classifier_state, batch)
                        logits = _logits[batch["marker"] == True]
                        logits = logits.reshape(-1, *_logits.shape[2:])
                        assert len(logits.shape) == 2 or len(logits.shape) == 4
                        logits_list.append(logits)
                logits = jnp.concatenate(logits_list, axis=0)
                shape = shape or logits.shape[1:]
                if shape:
                    assert logits.shape[1:] == shape
                with open(advfeature_path, "wb") as f:
                    logits = np.array(logits)
                    np.save(f, logits)
                del logits_list

            # valid set
            feature_path = f"{dir}/valid_features_M{mode_idx}S{i}.npy"
            if not os.path.exists(feature_path):
                valid_loader = dataloaders["val_loader"](rng=None)
                valid_loader = jax_utils.prefetch_to_device(
                    valid_loader, size=2)
                logits_list = []
                for batch_idx, batch in enumerate(valid_loader, start=1):
                    _logits = p_get_logits(classifier_state, batch)
                    logits = _logits[batch["marker"] == True]
                    logits = logits.reshape(-1, *_logits.shape[2:])
                    assert len(logits.shape) == 2 or len(logits.shape) == 4
                    logits_list.append(logits)
                logits = jnp.concatenate(logits_list, axis=0)
                shape = shape or logits.shape[1:]
                if shape:
                    assert logits.shape[1:] == shape
                with open(feature_path, "wb") as f:
                    logits = np.array(logits)
                    np.save(f, logits)
                del logits_list
            else:
                advfeature_path = f"{dir}/valid_advfeatures_M{mode_idx}S{i}.npy"
                valid_loader = dataloaders["val_loader"](rng=None)
                valid_loader = jax_utils.prefetch_to_device(
                    valid_loader, size=2)
                logits_list = []
                for batch_idx, batch in enumerate(valid_loader, start=1):
                    valid_rng = jax.random.fold_in(rng, batch_idx)
                    valid_rng = jax_utils.replicate(valid_rng)
                    batch = adv_attack(classifier_state, batch, valid_rng)
                    _logits = p_get_logits(classifier_state, batch)
                    logits = _logits[batch["marker"] == True]
                    logits = logits.reshape(-1, *_logits.shape[2:])
                    assert len(logits.shape) == 2 or len(logits.shape) == 4
                    logits_list.append(logits)
                logits = jnp.concatenate(logits_list, axis=0)
                shape = shape or logits.shape[1:]
                if shape:
                    assert logits.shape[1:] == shape
                with open(advfeature_path, "wb") as f:
                    logits = np.array(logits)
                    np.save(f, logits)
                del logits_list

            # test set
            feature_path = f"{dir}/test_features_M{mode_idx}S{i}.npy"
            if not os.path.exists(feature_path):
                test_loader = dataloaders["tst_loader"](rng=None)
                test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
                logits_list = []
                for batch_idx, batch in enumerate(test_loader, start=1):
                    _logits = p_get_logits(classifier_state, batch)
                    logits = _logits[batch["marker"] == True]
                    logits = logits.reshape(-1, *_logits.shape[2:])
                    assert len(logits.shape) == 2 or len(logits.shape) == 4
                    logits_list.append(logits)
                logits = jnp.concatenate(logits_list, axis=0)
                shape = shape or logits.shape[1:]
                if shape:
                    assert logits.shape[1:] == shape
                with open(feature_path, "wb") as f:
                    logits = np.array(logits)
                    np.save(f, logits)
                del logits_list
            else:
                advfeature_path = f"{dir}/test_advfeatures_M{mode_idx}S{i}.npy"
                test_loader = dataloaders["tst_loader"](rng=None)
                test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
                logits_list = []
                for batch_idx, batch in enumerate(test_loader, start=1):
                    test_rng = jax.random.fold_in(rng, batch_idx)
                    test_rng = jax_utils.replicate(test_rng)
                    batch = adv_attack(classifier_state, batch, test_rng)
                    _logits = p_get_logits(classifier_state, batch)
                    logits = _logits[batch["marker"] == True]
                    logits = logits.reshape(-1, *_logits.shape[2:])
                    assert len(logits.shape) == 2 or len(logits.shape) == 4
                    logits_list.append(logits)
                logits = jnp.concatenate(logits_list, axis=0)
                shape = shape or logits.shape[1:]
                if shape:
                    assert logits.shape[1:] == shape
                with open(advfeature_path, "wb") as f:
                    logits = np.array(logits)
                    np.save(f, logits)
                del logits_list
