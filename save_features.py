from tqdm import tqdm
from flax import jax_utils
from giung2.data.build import build_dataloaders
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from giung2.models.layers import FilterResponseNorm
from models.resnet import FlaxResNet
import sghmc
from functools import partial
import flax
from easydict import EasyDict
import defaults_sghmc as defaults
import os
import numpy as np
from utils import normalize, model_list, logit_dir_list, feature_dir_list


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

    # initialize model
    def initialize_model(key, model):
        @jax.jit
        def init(*args):
            return model.init(*args)
        return init({'params': key}, jnp.ones(dataloaders['image_shape'], model_dtype))
    variables = initialize_model(jax.random.PRNGKey(sghmc_config.seed), model)

    sghmc_state = sghmc.get_sghmc_state(
        sghmc_config, dataloaders, model, variables)
    sghmc_ckpt["model"] = sghmc_state
    # sghmc_ckpt["ckpt_dir"] = sghmc_ckpt_dir
    return sghmc_config, sghmc_ckpt


def load_classifer_state(mode_idx, sample_idx, template):
    sghmc_ckpt_dir = model_list[mode_idx]
    if not sghmc_ckpt_dir.endswith("sghmc"):
        sghmc_ckpt_dir = os.path.join(sghmc_ckpt_dir, "sghmc")
    sghmc_ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=sghmc_ckpt_dir, target=template, step=sample_idx
    )
    state = sghmc_ckpt["model"]
    return state, sghmc_ckpt


if __name__ == "__main__":
    """
    'logits' and 'features' are used compatibly as terms of any features of a given data
    """
    config, ckpt = get_ckpt_temp(model_list[0])
    n_samples_each_mode = 30
    n_modes = len(model_list)
    dataloaders = build_dataloaders(config)
    dir = "features_fixed"

    def get_logits(state, batch, feature_name="feature.vector"):
        x = batch["images"]
        # x = normalize(x)
        marker = batch["marker"]
        _, new_model_state = state.apply_fn({
            "params": state.params,
            "image_stats": state.image_stats
        }, x, rngs=None, mutable="intermediates")
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

    for mode_idx in range(n_modes):
        for i in tqdm(range(n_samples_each_mode)):
            feature_path = f"{dir}/train_features_M{mode_idx}S{i}.npy"
            # image_path = f"{dir}/train_images_M{mode_idx}S{i}.npy"
            if os.path.exists(feature_path):
                continue

            classifier_state, _ = load_classifer_state(mode_idx, i+1, ckpt)
            classifier_state = jax_utils.replicate(classifier_state)
            # train set
            train_loader = dataloaders["trn_loader"](rng=None)
            train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
            logits_list = []
            # images_list = []
            for batch_idx, batch in enumerate(train_loader, start=1):
                shape = batch["images"].shape
                _logits = p_get_logits(classifier_state, batch)
                logits = _logits[batch["marker"] ==
                                 True].reshape(-1, _logits.shape[-1])
                assert len(logits.shape) == 2
                # images = batch["images"][batch["marker"]
                #                          == True].reshape(-1, *shape[2:])
                # assert len(images.shape) == 4
                logits_list.append(logits)
                # images_list.append(images)
            logits = jnp.concatenate(logits_list, axis=0)
            # images = jnp.concatenate(images_list, axis=0)
            with open(feature_path, "wb") as f:
                logits = np.array(logits)
                np.save(f, logits)
            # with open(image_path, "wb") as f:
            #     images = np.array(images)
            #     np.save(f, images)
            del logits_list
            # del images_list

            # valid set
            valid_loader = dataloaders["val_loader"](rng=None)
            valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
            logits_list = []
            # images_list = []
            for batch_idx, batch in enumerate(valid_loader, start=1):
                shape = batch["images"].shape
                _logits = p_get_logits(classifier_state, batch)
                logits = _logits[batch["marker"] ==
                                 True].reshape(-1, _logits.shape[-1])
                assert len(logits.shape) == 2
                # images = batch["images"][batch["marker"]
                #                          == True].reshape(-1, *shape[2:])
                # assert len(images.shape) == 4
                logits_list.append(logits)
                # images_list.append(images)
            logits = jnp.concatenate(logits_list, axis=0)
            # images = jnp.concatenate(images_list, axis=0)
            with open(f"{dir}/valid_features_M{mode_idx}S{i}.npy", "wb") as f:
                logits = np.array(logits)
                np.save(f, logits)
            # with open(f"{dir}/valid_images_M{mode_idx}S{i}.npy", "wb") as f:
            #     images = np.array(images)
            #     np.save(f, images)
            del logits_list
            # del images_list
            # test set
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            logits_list = []
            # images_list = []
            for batch_idx, batch in enumerate(test_loader, start=1):
                shape = batch["images"].shape
                _logits = p_get_logits(classifier_state, batch)
                logits = _logits[batch["marker"] ==
                                 True].reshape(-1, _logits.shape[-1])
                assert len(logits.shape) == 2
                # images = batch["images"][batch["marker"]
                #                          == True].reshape(-1, *shape[2:])
                # assert len(images.shape) == 4
                logits_list.append(logits)
                # images_list.append(images)
            logits = jnp.concatenate(logits_list, axis=0)
            # images = jnp.concatenate(images_list, axis=0)
            with open(f"{dir}/test_features_M{mode_idx}S{i}.npy", "wb") as f:
                logits = np.array(logits)
                np.save(f, logits)
            # with open(f"{dir}/test_images_M{mode_idx}S{i}.npy", "wb") as f:
            #     images = np.array(images)
            #     np.save(f, images)
            del logits_list
            # del images_list
