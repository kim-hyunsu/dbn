# Import numpy
import numpy as np

# Import scikitlearn for machine learning functionalities
from sklearn.manifold import TSNE

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt

import seaborn as sb
from giung2.models.layers import FilterResponseNorm
# from giung2.models.resnet import FlaxResNet
from models.resnet import FlaxResNet
from giung2.data.build import build_dataloaders
import defaults_sghmc as defaults
from flax.training import checkpoints, train_state
from flax import jax_utils
import flax
import jax.numpy as jnp
import jax
from functools import partial
import os
from easydict import EasyDict
from typing import Any
import sghmc_deprecated


class TrainStateBatch(train_state.TrainState):
    image_stats: Any
    batch_stats: Any


class TrainState(train_state.TrainState):
    image_stats: Any


def plot(x, colors):

    palette = np.array(sb.color_palette("hls", 10))  # Choosing color palette

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors])
    txts = []
    # for i in range(10):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
    #     txts.append(txt)
    plt.savefig("tSNE2.pdf", dpi=300)
    return f, ax, txts


def evaluate(ckpt, image):
    state = ckpt["model"]
    _, new_model_state = state.apply_fn({
        "params": state.params,
        "image_stats": state.image_stats
    }, image,
        rngs=None,
        mutable=["intermediates"])

    logits = new_model_state["intermediates"]["cls.logit"][0]
    return logits


ckpt_list = [
    "./checkpoints/frn_sd2",
    "./checkpoints/frn_sd3",
    "./checkpoints/frn_sd5",
    "./checkpoints/frn_sd7",
    "./checkpoints/frn_sd11",
    "./checkpoints/frn_sd13",
    "./checkpoints/frn_sd17"
]

if __name__ == "__main__":
    sgd_ckpt_dir = ckpt_list[0]
    sghmc_ckpt_dir = os.path.join(ckpt_list[0], "sghmc")
    sgd_ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=sgd_ckpt_dir, target=None)
    sgd_config = EasyDict(sgd_ckpt["config"])
    sghmc_ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=sghmc_ckpt_dir, target=None)
    sghmc_config = EasyDict(sghmc_ckpt["config"])
    dataloaders = build_dataloaders(sgd_config)
    # loader = dataloaders["val_loader"](rng=jax.random.PRNGKey(2023))
    loader = dataloaders["trn_loader"](rng=jax.random.PRNGKey(2023))
    loader = jax_utils.prefetch_to_device(loader, size=2)
    batch = next(loader)
    batch = next(loader)
    del loader
    del sgd_ckpt["model"]
    del sghmc_ckpt["model"]
    image = batch["images"][0][:1]

    # specify precision
    model_dtype = jnp.float32
    if sgd_config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build model
    _ResNet = partial(
        FlaxResNet,
        depth=sgd_config.model_depth,
        widen_factor=sgd_config.model_width,
        dtype=model_dtype,
        pixel_mean=defaults.PIXEL_MEAN,
        pixel_std=defaults.PIXEL_STD,
        num_classes=dataloaders['num_classes'])

    if sgd_config.model_style == 'BN-ReLU':
        model = _ResNet()
    elif sgd_config.model_style == "FRN-Swish":
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
        return init({'params': key}, jnp.ones(dataloaders['image_shape'], model.dtype))
    variables = initialize_model(jax.random.PRNGKey(sgd_config.seed), model)

    sgd_state = sghmc_deprecated.get_sgd_state(sgd_config, dataloaders, model, variables)
    sghmc_state = sghmc_deprecated.get_sghmc_state(
        sghmc_config, dataloaders, model, variables)
    samples = []
    indices = []
    mode_idx = []
    for m, ckpt_dir in enumerate(ckpt_list):
        samples_each_mode = []
        idx_each_mode = []
        sgd_ckpt["model"] = sgd_state
        sgd_ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir, target=sgd_ckpt)
        feat = evaluate(sgd_ckpt, image)
        samples_each_mode.append(feat[0])
        idx_each_mode.append(m)
        ckpt_dir = os.path.join(ckpt_dir, "sghmc")
        for idx in range(1, 1001):
            try:
                sghmc_ckpt["model"] = sghmc_state
                sghmc_ckpt = checkpoints.restore_checkpoint(
                    ckpt_dir=ckpt_dir, target=sghmc_ckpt, step=idx)
                feat = evaluate(sghmc_ckpt, image)
                if idx == 0:
                    print("shape of logits", feat.shape)
                samples_each_mode.append(feat[0])
                idx_each_mode.append(m)
            except Exception as e:
                print(e)
                break
        samples_each_mode = jnp.array(samples_each_mode)
        idx_each_mode = jnp.array(idx_each_mode)
        samples.append(samples_each_mode)
        mode_idx.append(idx_each_mode)
        indices.append(len(samples_each_mode))

    indices.pop()
    temp = jnp.concatenate(samples, axis=0)
    labels = jnp.concatenate(mode_idx, axis=0)
    reducted = TSNE(perplexity=30).fit_transform(temp)

    plot(reducted, labels)
