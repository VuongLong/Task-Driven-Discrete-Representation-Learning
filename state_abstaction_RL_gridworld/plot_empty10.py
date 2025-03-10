import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from visgrid.utils import load_experiment, get_parser

parser = get_parser()
# yapf: disable
parser.add_argument('--pretrain-steps', type=str, default='10000', # change 3k, 10k, 30k
                     help='Number of pretraining steps')
parser.add_argument('--smoothing', type=int, default=10,
                    help='Number of data points for sliding window average')
# yapf: enable
args = parser.parse_args()

walls = "empty"


def load_experiment(path):
    logfiles = sorted(glob.glob(os.path.join(path, "scores-*.txt")))
    print("logfiles", logfiles)
    agents = [path.split("/")[-2] for f in logfiles]
    seeds = [int(f.split("-")[-1].split(".")[0]) for f in logfiles]
    logs = [open(f, "r").read().splitlines() for f in logfiles]

    def read_log(log):
        results = [json.loads(item) for item in log]
        data = smooth(pd.DataFrame(results), args.smoothing)
        return data

    results = [read_log(log) for log in logs]
    keys = list(zip(agents, seeds))
    data = (
        pd.concat(results, join="outer", keys=keys, names=["agent", "seed"])
        .sort_values(by="seed", kind="mergesort")
        .reset_index(level=[0, 1])
    )
    return data  # [data['episode']<=100]


def smooth(data, n):
    numeric_dtypes = data.dtypes.apply(pd.api.types.is_numeric_dtype)
    numeric_cols = numeric_dtypes.index[numeric_dtypes]
    data[numeric_cols] = data[numeric_cols].rolling(n).mean()
    return data


pretrain_experiments = f"pretrain_empty10_10000"  # change in pretrain_empty10_newseed, pretrain_empty10_3000, pretrain_empty10_10000
experiments = [pretrain_experiments]

steps = "1000" if args.pretrain_steps == "1k" else args.pretrain_steps

list_names = [
    f"{steps}_continuous_phi_10_{walls}",
    f"{steps}_quantize_phi_10_500_{walls}",
    f"{steps}_quantize_phi_10_100_{walls}",
    f"{steps}_quantize_phi_10_50_{walls}",
]

lables_all = [
    "Continuous",
    "$|C|=500$",
    "$|C|=100$",
    "$|C|=50$",
    # "VQ-WAE 16",
    "VQ-VAE 500",
    "VQ-VAE 100",
    "VQ-VAE 50",
    # "VQ-VAE 16",
    "SQ-VAE 500",
    "SQ-VAE 100",
    "SQ-VAE 50",
    # "VQ-SAE 16",
][: len(list_names)]

COLOR_LIST = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

# agents = [
#     # f"{steps}_continuous_phi",
#     # f"{steps}_quantize_phi_16",
#     # f"{steps}_quantize_phi_32",
#     f"{steps}_quantize_phi_36",
#     # f"{steps}_quantize_phi_64",
#     # f"{steps}_quantize_phi_128",
#     # 'inv-only',
#     # 'contr-only',
#     # 'autoenc',
#     # 'truestate',
#     # 'end-to-end',
#     # 'pixel-pred',
#     # 'random',
#     # 'rearrange_xy',
# ]
agents = list_names
root = "results/scores/"
unfiltered_paths = [
    (root + e + "/" + a + "/", (e, a)) for e in experiments for a in agents
]
experiments = [
    experiment for path, experiment in unfiltered_paths if os.path.exists(path)
]

print("=" * 100)
print("unfiltered path")
for p in unfiltered_paths:
    print(p)
print("=" * 100)
paths = [path for path, _ in unfiltered_paths if os.path.exists(path)]

print("-" * 50)
print("filted paths:")
for p in paths:
    print(p)
print("-" * 50)

# print([load_experiment(p) for p in paths])
labels = ["tag", "features"]
data = pd.concat(
    [load_experiment(p) for p in paths], join="outer", keys=(experiments), names=labels
).reset_index(level=list(range(len(labels))))


def plot(data, x, y, hue, style, col=None):
    print("Plotting using hue={hue}, style={style}".format(hue=hue, style=style))
    assert not data.empty, "DataFrame is empty, please check query"

    # print(data.query('episode==99').groupby('agent', as_index=False)['total_reward'].mean())
    # print(data.query('episode==99').groupby('agent', as_index=False)['total_reward'].std())

    data = data.replace("markov", "Markov")
    data = data.replace("quantize", "Quantize")
    data = data.replace("end-to-end", "visual")
    data = data.replace("truestate", "xy-position")

    # print(data.groupby("agent", as_index=False)["reward"].mean())
    # print(data.groupby("agent", as_index=False)["reward"].std())

    # If asking for multiple envs, use facetgrid and adjust height
    height = 4 if col is not None and len(data[col].unique()) > 1 else 5
    if col:
        col_wrap = 2 if len(data[col].unique()) > 1 else 1
    else:
        col_wrap = None

    # data = data[data['episode'] < 97]

    # dashes_list = [
    #     "",
    #     (1, 1),
    #     (1, 2, 5, 2),
    #     (2, 2, 1, 2),
    #     (2, 2, 1, 2),
    #     (7, 2, 3, 2),
    #     (3, 2, 3, 2),
    #     (7, 1, 1, 1),
    # ]
    dashes_list = (
        [""] * 4
        + [(1, 1)] * 3
        + [((3, 2, 5, 2))] * 3
        + [
            (1, 1),
            (1, 2, 5, 2),
            (3, 2, 5, 2),
            (2, 2, 1, 2),
            (3, 2, 1, 2),
        ]
        * 10
    )
    dashes_list = dashes_list[: len(list_names)]
    dashes = {name: d for name, d in zip(list_names, dashes_list)}

    # dashes = {
    #     # f"{steps}_continuous_phi_{walls}": "",
    #     # f"{steps}_quantize_phi_16_{walls}": (1, 1),
    #     # f"{steps}_quantize_phi_32_{walls}": (1, 1),
    #     f"{steps}_quantize_phi_36_{walls}": "",
    #     # f"{steps}_quantize_phi_64_{walls}": (1, 1),
    #     # f"{steps}_quantize_phi_128_{walls}": (1, 2, 5, 2),
    #     # 'inv-only': (1, 1),
    #     # 'contr-only': (1, 2, 5, 2),
    #     # 'autoenc': (2, 2, 1, 2),
    #     # 'visual': (2, 2, 1, 2),
    #     # 'xy-position': (7, 2, 3, 2),
    #     # 'pixel-pred': (7, 1, 1, 1),
    #     # "random": (1, 2, 3, 2),
    # }
    algs = list_names
    # algs = [
    #     # f"{steps}_continuous_phi_{walls}",
    #     # f"{steps}_quantize_phi_16_{walls}",
    #     # f"{steps}_quantize_phi_32_{walls}",
    #     f"{steps}_quantize_phi_36_{walls}",
    #     # f"{steps}_quantize_phi_64_{walls}",
    #     # f"{steps}_quantize_phi_128_{walls}",
    #     # 'autoenc',
    #     # 'inv-only',
    #     # 'pixel-pred',
    #     # 'contr-only',
    #     # 'visual',
    #     # 'xy-position',
    #     # "random",
    # ]
    labels = list_names
    labels = lables_all
    # labels = [
    #     # f"{steps}_continuous_phi_{walls}",
    #     # f"{steps}_quantize_phi_16_{walls}",
    #     # f"{steps}_quantize_phi_32_{walls}",
    #     f"{steps}_quantize_phi_36_{walls}",
    #     # f"{steps}_quantize_phi_64_{walls}",
    #     # f"{steps}_quantize_phi_128_{walls}",
    #     # 'Autoenc',
    #     # 'Inverse',
    #     # 'Pixel-Pred',
    #     # 'Ratio',
    #     # 'Visual',
    #     # 'Expert (x,y)',
    #     # "Random",
    # ]
    colormap = list_names
    # colormap = [
    #     # f"{steps}_continuous_phi_{walls}",
    #     # f"{steps}_quantize_phi_16_{walls}",
    #     # f"{steps}_quantize_phi_32_{walls}",
    #     f"{steps}_quantize_phi_36_{walls}",
    #     # f"{steps}_quantize_phi_64_{walls}",
    #     # f"{steps}_quantize_phi_128_{walls}",
    #     # 'inv-only',
    #     # 'autoenc',
    #     # 'visual',
    #     # 'contr-only',
    #     # 'xy-position',
    #     # "pixel-pred",
    # ]
    p = sns.color_palette("Set1", n_colors=2)
    red, _ = p

    p = sns.color_palette("Set1", n_colors=9, desat=0.5)
    _, blue, green, purple, orange, yellow, brown, pink, gray = p

    palette = [red, green, blue, brown, purple, orange, yellow, pink][: len(labels) - 1]
    palette = [
        red,
        green,
        blue,
        brown,
        purple,
        orange,
        yellow,
        pink,
        gray,
        *COLOR_LIST,
    ][: len(labels)]
    palette = dict(zip(colormap, palette))
    # palette["random"] = gray
    # data = data.append(
    #     {"agent": "random", "reward": -84.8, "seed": 0, "episode": 0}, ignore_index=True
    # )  # yapf: disable

    g = sns.relplot(
        x=x,
        y=y,
        data=data,
        hue=hue,
        hue_order=algs,
        style=style,
        kind="line",
        # legend='full',
        legend=False,
        dashes=dashes,
        height=height,
        aspect=1.2,
        col=col,
        col_wrap=col_wrap,
        # col_order=col_order,
        palette=palette,
        linewidth=2,
        # alpha=0.1,
        facet_kws={"sharey": False, "sharex": False},
    )

    g.set_titles("{col_name}")

    ax = g.axes.flat[0]
    ax.set_ylim((-250, 0))
    # ax.set_xlim((0, 100))
    # ax.axhline(-84.8, dashes=dashes["random"], color=palette["random"], linewidth=2)
    leg = ax.legend(
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.4, -0.17),
        fontsize=10,
        frameon=False,
    )
    leg.set_draggable(True)
    for axis in ["bottom", "left"]:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)
    ax.tick_params(labelsize=16)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_xlabel("Episode", fontsize=12)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.25)
    plt.savefig(f"final_{steps}_all_{args.pretrain_steps}_{walls}10.pdf")
    plt.show()


plot(data, x="episode", y="reward", hue="agent", style="agent")
