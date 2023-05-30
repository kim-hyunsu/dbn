
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from matplotlib import pyplot as plt

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


def cls_plot(A, B, C, labels, y, verbose, epoch=None, div="test"):
    nrows = 1
    nlabels = A.shape[-1]
    fig, ax = plt.subplots(ncols=1, nrows=nrows, figsize=(
        6, 2*(1.6 * nrows + 0.3)), squeeze=False)

    if acc is not None:
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
