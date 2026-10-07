from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_COLOR = "#D97745"
KD_COLOR = "#167D8D"


def plot_benchmark(results, x_key, xlabel, title, filename,
                   log_x=False, categorical=False):
    values = [row[x_key] for row in results]
    x = np.arange(len(values)) if categorical else np.array(values)
    base_time = np.array([row["base_time"] for row in results]) * 1000
    kd_time = np.array([row["kd_time"] for row in results]) * 1000
    calculations_fraction = np.array([
        row["kd_calculations"] / row["base_calculations"] for row in results
    ])
    speedup = base_time / kd_time

    figure, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    figure.suptitle(title, fontsize=16)

    axes[0].plot(x, base_time, "o-", color=BASE_COLOR, label="Полный перебор")
    axes[0].plot(x, kd_time, "o-", color=KD_COLOR, label="KD-дерево")
    axes[0].set(xlabel=xlabel, ylabel="мс на запрос", title="Время поиска")
    axes[0].legend()

    axes[1].plot(x, calculations_fraction, "o-", color=KD_COLOR)
    axes[1].set(
        xlabel=xlabel,
        ylabel="доля от полного перебора",
        title="Вычисленные расстояния",
    )

    axes[2].plot(x, speedup, "o-", color=KD_COLOR)
    axes[2].axhline(1, color=BASE_COLOR, linestyle="--")
    axes[2].set(
        xlabel=xlabel,
        ylabel="время перебора / время KD",
        title="Ускорение",
    )

    for axis in axes:
        axis.grid(alpha=0.3)
        if log_x:
            axis.set_xscale("log")
        if categorical:
            axis.set_xticks(x, values, rotation=25, ha="right")

    output = Path(__file__).resolve().parents[1] / "images" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    return output
