# Import numpy
import numpy as np
from tqdm import tqdm

# Import scikitlearn for machine learning functionalities
from sklearn.manifold import TSNE

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt

import seaborn as sb
from giung2.models.layers import FilterResponseNorm
from dsb import build_featureloaders
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
from utils import model_list
import matplotlib.patheffects as pe


class TrainStateBatch(train_state.TrainState):
    image_stats: Any
    batch_stats: Any


class TrainState(train_state.TrainState):
    image_stats: Any


def main():
    dir = "features_fixed"
    mixup_dir = "features_1mixup10_fixed"
    settings = __import__(f"{dir}.settings", fromlist=[""])
    data_name = settings.data_name
    model_style = settings.model_style
    ckpt_list = model_list(data_name, model_style)
    # samples of the first mode (sghmc)
    sghmc_ckpt_dir = os.path.join(ckpt_list[0], "sghmc")
    # sghmc checkpoint
    sghmc_ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=sghmc_ckpt_dir, target=None)
    sghmc_config = EasyDict(sghmc_ckpt["config"])
    del sghmc_ckpt["model"]

    label_dict = {
        "train": 0,
        "valid": 1,
        "mixup_train": 2,
        "mixup_valid": 3,
    }

    # ----------------------------------------------------------------
    # take some data
    # ----------------------------------------------------------------
    logitsB_list = []
    logitsA_list = []
    labels_list = []

    def stack_logits(div, loaders, normalize, get_data, max_len=None):
        loaders = jax_utils.prefetch_to_device(loaders, size=2)
        for batch_idx, batch in enumerate(tqdm(loaders), start=1):
            batch_size = batch["images"].shape[0] * batch["images"].shape[1]
            if max_len is not None and batch_idx*batch_size >= max_len:
                break
            logitsB, _ = get_data(batch, "images")
            logitsA, y = get_data(batch, "labels")
            logitsB = logitsB[y == 3].reshape(-1, logitsB.shape[-1])
            logitsA = logitsA[y == 3].reshape(-1, logitsA.shape[-1])
            marker = batch["marker"][y == 3].reshape(-1, 1)
            logitsB = jnp.where(marker, normalize(logitsB), 0)
            logitsA = jnp.where(marker, normalize(logitsA), 0)
            batch_size = logitsB.shape[0]
            labels = jnp.ones((batch_size,)) * label_dict[div]
            logitsB_list.append(logitsB)
            logitsA_list.append(logitsA)
            labels_list.append(labels)

    sghmc_config.features_dir = dir
    sghmc_config.n_Amodes = 1
    sghmc_config.n_samples_each_mode = 1
    sghmc_config.take_valid = False
    sghmc_config.get_stats = False
    floaders, normalize, unnormalize, get_data = build_featureloaders(
        sghmc_config)
    max_len = None
    loaders = floaders["trn_featureloader"](rng=None)
    stack_logits("train", loaders, normalize, get_data, max_len=max_len)
    loaders = floaders["val_featureloader"](rng=None)
    stack_logits("valid", loaders, normalize, get_data, max_len=max_len)

    sghmc_config.features_dir = mixup_dir
    floaders, normalize, unnormalize, get_data = build_featureloaders(
        sghmc_config)
    loaders = floaders["trn_featureloader"](rng=None)
    stack_logits("mixup_train", loaders, normalize, get_data, max_len=max_len)
    # loaders = floaders["val_featureloader"](rng=None)
    # stack_logits("mixup_valid", loaders, normalize, get_data, max_len=max_len)

    # ----------------------------------------------------------------
    # take the train state
    # ----------------------------------------------------------------
    tempB = jnp.concatenate(logitsB_list, axis=0)
    tempA = jnp.concatenate(logitsA_list, axis=0)
    labels = jnp.concatenate(labels_list, axis=0, dtype=jnp.int32)
    with open("tsne/labels_fixed.npy", "wb") as f:
        np.save(f, labels)
    reductedB = TSNE(perplexity=50, n_jobs=128, verbose=1).fit_transform(tempB)
    with open("tsne/reductedB_fixed.npy", "wb") as f:
        np.save(f, reductedB)
    reductedA = TSNE(perplexity=50, n_jobs=128, verbose=1).fit_transform(tempA)
    with open("tsne/reductedA_fixed.npy", "wb") as f:
        np.save(f, reductedA)


def load_and_plot():
    def plot(xs, colors):
        # palette = np.array(sb.color_palette("hls", 10)
        #                    )  # Choosing color palette
        palette = np.array(["green", "red", "blue", "black",
                            "midnightblue", "darkviolet", "deeppink", "violet", "indigo", "slategray"])
        size = np.array([6, 6, 2, 2, 2, 2, 2, 2, 2, 2])
        # f = plt.figure(figsize=(8, 8))
        # ax = plt.subplot(aspect='equal')
        f, axs = plt.subplots(1, 2, figsize=(16, 8))
        for i, x in enumerate(xs):
            axs[i].set_aspect('equal')
            axs[i].scatter(x[:, 0], x[:, 1], lw=0,
                           s=size[colors], c=palette[colors])
        txts = []
        # for i in range(10):
        #     # Position of each label.
        #     xtext, ytext = np.median(x[colors == i, :], axis=0)
        #     if xtext == float("nan"):
        #         continue
        #     for ax in axs:
        #         txt = ax.text(xtext, ytext, str(i), fontsize=24)
        #     txt.set_path_effects(
        #         [pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        #     txts.append(txt)
        plt.savefig("tsne/tSNE_dataset_fixed.pdf", dpi=300)
        return f, axs, txts
    print("Loading...")
    with open("tsne/reductedB_fixed.npy", "rb") as f:
        reductedB = np.load(f)
    with open("tsne/reductedA_fixed.npy", "rb") as f:
        reductedA = np.load(f)
    with open("tsne/labels_fixed.npy", "rb") as f:
        labels = np.load(f)
    plot([reductedB, reductedA], labels)
    print("Plotted")


if __name__ == "__main__":
    main()
    load_and_plot()
