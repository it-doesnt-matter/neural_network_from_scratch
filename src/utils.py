import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import network
from layer import Layer
from rocket_color_map import register_additional_cmaps


# points is the amount of points PER CLASS
def generate_spiral_data(
    points: int = 100,
    classes: int = 3,
    train_size: float = 0.8,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    data = np.zeros((points * classes, 2))
    target = np.zeros(points * classes, dtype="uint8")
    for i in range(classes):
        radius = np.linspace(10.0, 1, points)
        theta = np.linspace(i * 8, (i + 1) * 9, points) + rng.standard_normal(points) * 0.25
        index_range = range(points * i, points * (i + 1))
        data[index_range] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        target[index_range] = i
    data_train, data_test, target_train, target_test = train_test_split(
        data, target, train_size=train_size, stratify=target
    )
    if plot:
        colors = np.array(["r", "g", "b"])
        plt.scatter(data[:, 0], data[:, 1], c=colors[target].tolist(), s=30)
        plt.show()
    return (data_train, data_test, target_train, target_test)


def shuffle_data(data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = random.sample(range(len(data)), len(data))
    data = np.array(list(map(data.__getitem__, indices)))
    target = np.array(list(map(target.__getitem__, indices)))
    return (data, target)


def draw_decision_boundary(
    ax: any,
    network: network.SequentialNetwork,
    data: np.ndarray,
    target: np.ndarray,
) -> None:
    h = 0.02
    x_min = data[:, 0].min() - 1
    x_max = data[:, 0].max() + 1
    y_min = data[:, 1].min() - 1
    y_max = data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    colors = np.array(["r", "g", "b"])
    ax.scatter(data[:, 0], data[:, 1], color=colors[target].tolist(), s=100, alpha=0.9, zorder=2)
    prediction = network.predict(mesh_data, 64)
    prediction = prediction.reshape(xx.shape)
    ax.contourf(xx, yy, prediction, colors=("r", "g", "b"), levels=2, alpha=0.4, zorder=1)


# didn't use row labels, cause they were causing layout problems
def draw_summary_table(ax: any, acc: float, loss: float, epochs: int, batch_size: int) -> None:
    col_labels = ["", "final stats"]
    cell_text = [
        ["loss", loss],
        ["accuracy", acc],
        ["epochs", epochs],
        ["batch size", batch_size],
    ]

    summary_table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
    )

    summary_table[0, 0].visible_edges = ""
    summary_table[0, 1].set(linewidth=2)
    summary_table[0, 1].set_text_props(fontweight="bold")
    for i in range(1, 5):
        summary_table[i, 0].set(linewidth=2)
        summary_table[i, 0].set_text_props(ha="left", fontweight="bold")


def draw_architecture_table(ax: any, arch: list[tuple[Layer, int]]) -> None:
    col_labels = ["Layers", "# Neurons"]
    cell_text = []
    for name, size in arch:
        cell_text.append((name, size))

    arch_table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
    )

    arch_table[0, 0].set(linewidth=2)
    arch_table[0, 0].set_text_props(fontweight="bold")
    arch_table[0, 1].set(linewidth=2)
    arch_table[0, 1].set_text_props(fontweight="bold")


def draw_optimizer_and_loss_table(
    ax: any,
    optimizer_info: dict[str, str | float | int],
    loss_name: str,
) -> None:
    col_labels = ["Loss", loss_name]
    cell_text = [
        ["", ""],
        ["", "Optimizer"],
    ]
    for key, value in optimizer_info.items():
        cell_text.append([key, value])

    opt_loss_table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
    )

    opt_loss_table[0, 0].set(linewidth=2)
    opt_loss_table[0, 0].set_text_props(ha="left", fontweight="bold")
    opt_loss_table[1, 0].visible_edges = ""
    opt_loss_table[1, 1].visible_edges = ""
    opt_loss_table[2, 0].visible_edges = ""
    opt_loss_table[2, 1].set(linewidth=2)
    opt_loss_table[2, 1].set_text_props(ha="center", fontweight="bold")
    amount_rows = len(optimizer_info.keys())
    for i in range(3, 3 + amount_rows):
        opt_loss_table[i, 0].set(linewidth=2)
        opt_loss_table[i, 0].set_text_props(ha="left", fontweight="bold")


def draw_training_summary(
    network: network.SequentialNetwork,
    data: np.ndarray,
    target: np.ndarray,
    summary: dict[str, list[float]],
    optimizer_info: dict[str, str | float | int],
    loss_name: str,
    file_name: str = "",
) -> None:
    set_mpl_theme()

    _, axs = plt.subplot_mosaic([
        ["dec", "loss", "summary"],
        ["dec", "acc", "arch"],
        ["dec", "rate", "opt_loss"],
    ])

    draw_decision_boundary(axs["dec"], network, data, target)

    axs["loss"].plot(summary["training_loss"])
    axs["loss"].plot(summary["validation_loss"], alpha=0.5)
    axs["loss"].set_title("Loss")

    axs["acc"].plot(summary["training_accuracy"])
    axs["acc"].plot(summary["validation_accuracy"], alpha=0.5)
    axs["acc"].set_title("Accuracy")

    axs["rate"].plot(summary["learning_rate"])
    axs["rate"].set_title("Rate")

    if summary["validation_loss"] is not None and summary["validation_accuracy"] is not None:
        draw_summary_table(
            axs["summary"],
            f"{summary['validation_accuracy'][-1]:.2%}",
            f"{summary['validation_loss'][-1]:.4}",
            summary["epochs"],
            summary["batch_size"],
        )
    else:
        draw_summary_table(
            axs["summary"],
            f"{summary['training_loss'][-1]:.4}",
            f"{summary['training_accuracy'][-1]:.2%}",
            summary["epochs"],
            summary["batch_size"],
        )
    axs["summary"].axis("off")

    draw_architecture_table(axs["arch"], network.get_architecture())
    axs["arch"].axis("off")

    draw_optimizer_and_loss_table(axs["opt_loss"], optimizer_info, loss_name)
    axs["opt_loss"].axis("off")

    if file_name:
        plt.savefig(file_name)
    plt.show()


def set_mpl_theme() -> None:
    register_additional_cmaps()

    mpl.rcParams["patch.edgecolor"] = "w"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["patch.force_edgecolor"] = True
    mpl.rcParams["text.color"] = "0.15"
    mpl.rcParams["axes.facecolor"] = "#eaeaf2"
    mpl.rcParams["axes.edgecolor"] = "#ffffff"
    mpl.rcParams["axes.linewidth"] = 1.25
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["axes.axisbelow"] = True
    mpl.rcParams["xtick.color"] = "0.15"
    mpl.rcParams["xtick.major.size"] = 6
    mpl.rcParams["xtick.minor.size"] = 4
    mpl.rcParams["xtick.major.width"] = 1.25
    mpl.rcParams["xtick.minor.width"] = 1
    mpl.rcParams["ytick.color"] = "0.15"
    mpl.rcParams["ytick.major.size"] = 6
    mpl.rcParams["ytick.minor.size"] = 4
    mpl.rcParams["ytick.major.width"] = 1.25
    mpl.rcParams["ytick.minor.width"] = 1
    mpl.rcParams["grid.color"] = "#ffffff"
    mpl.rcParams["grid.linewidth"] = 1
    mpl.rcParams["figure.figsize"] = (15, 5)
    mpl.rcParams["figure.constrained_layout.use"] = True
