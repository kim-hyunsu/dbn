
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

import yaml

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter, FuncFormatter
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
c1 = sns.color_palette("deep")[0]
c2 = sns.color_palette("deep")[1]
c3 = sns.color_palette("deep")[2]
label_names = np.array(['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                       'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'])
c_list = sns.color_palette("rocket_r")


def cls_plot(A, B, C, labels, y, verbose, epoch=None, div="test"):
    nrows = 1
    nlabels = A.shape[-1]
    fig, ax = plt.subplots(ncols=1, nrows=nrows, figsize=(
        6, 2*(1.6 * nrows + 0.3)), squeeze=False)

    if epoch is not None:
        ax[0, 0].set_title(
            f"pred of <{label_names[y]}> in {div} set at {epoch}")
    else:
        ax[0, 0].set_title(f"pred of <{label_names[y]}> in {div} set")

    for j in range(0, 1):
        ax[nrows - 1, j].set_xticks(
            np.arange(nlabels), labels=label_names[:len(labels)][labels], rotation=50)

    for i in range(nrows):

        ax[i, 0].set_ylabel("$p_{\\theta(r)}(y_i|x)$")
        ax[i, 0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax[i, 0].bar(np.arange(nlabels) - 0.2,   B, 0.2, color=c1, label="A")
        ax[i, 0].bar(np.arange(nlabels),   C, 0.2, color=c2, label="C")
        ax[i, 0].bar(np.arange(nlabels) + 0.2,   A, 0.2, color=c3, label="B")
        ax[i, 0].set_xlim(-0.5, nlabels-0.5)
        ax[i, 0].set_ylim(0, 1)
        ax[i, 0].tick_params(axis="y", labelsize=nlabels)

    handles, names = ax[0, 0].get_legend_handles_labels()

    fig.legend(handles, names, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"figures/{verbose}.pdf", bbox_inches="tight")
    plt.close(fig)


c10_label_names = ["plane", "auto", "bird", "cat",
                   "deer", "dog", "frog", "horse", "ship", "truck"]


def dbn_plot(probs, labels, images):
    nrows = 2
    idx_list = [0, 1]
    fig, ax = plt.subplots(ncols=4, nrows=nrows, figsize=(
        12, 1.6 * nrows + 0.3), squeeze=False)
    fig.rc("font", size=15)
    ax[0, 0].set_title("Images")
    ax[0, 1].set_title("Source")
    ax[0, 2].set_title("Diffusion Steps")
    ax[0, 3].set_title("Target")

    for j in range(1, 4):
        ax[nrows - 1, j].set_xticks(np.arange(10),
                                    labels=c10_label_names, rotation=50)

    for i in range(nrows):
        idx = idx_list[i]

        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 0].set_ylabel(c10_label_names[labels[idx]])
        ax[i, 1].set_ylabel("$Confidences$")
        ax[i, 1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax[i, 2].yaxis.set_major_formatter(NullFormatter())
        ax[i, 3].yaxis.set_major_formatter(NullFormatter())

        ax[i, 0].imshow(images[idx])
        ax[i, 1].bar(np.arange(10), probs[idx][0], 0.4,
                     color=c_list[0])
        for j in range(5):
            ax[i, 2].bar(np.arange(10) - 0.4 + 0.2*j, probs[idx][j+1],
                         0.2, color=c_list[j+1], label=f"t={0.2*(j+1):.1f}")
        ax[i, 3].bar(np.arange(10), probs[idx][-1], 0.4,
                     color=c_list[-1])

        for j in range(1, 4):
            if i < nrows - 1:
                ax[i, j].set_xticks(np.arange(10), labels=[""] * 10)
            ax[i, j].set_xlim(-0.5, 9.5)
            ax[i, j].set_ylim(0, 1)
            ax[i, j].tick_params(axis="y", labelsize=8)

    handles, names = ax[0, 2].get_legend_handles_labels()

    # fig.legend(handles, names, loc="lower center", ncol=4, bbox_to_anchor=(0.62, -0.07))
    fig.legend(handles, names, loc="center right", bbox_to_anchor=(1.09, 0.55))
    fig.tight_layout()
    fig.show()
    fig.savefig("sample_short.pdf", bbox_inches="tight")
