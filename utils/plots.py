from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_COLOR = "#D97745"
KD_COLOR = "#167D8D"
KERNEL_COLORS = (
    "#167D8D",
    "#D97745",
    "#725AC1",
    "#4C956C",
)


def plot_benchmark(results, x_key, xlabel, title, filename,
                   log_x=False, categorical=False):
    distributions = list(dict.fromkeys(row["distribution"] for row in results))
    figure, axes = plt.subplots(
        len(distributions),
        3,
        figsize=(16, 4 * len(distributions)),
        squeeze=False,
        layout="constrained",
    )
    figure.suptitle(title, fontsize=16)

    for row_index, distribution in enumerate(distributions):
        rows = [row for row in results if row["distribution"] == distribution]
        values = [row[x_key] for row in rows]
        x = np.arange(len(values)) if categorical else np.array(values)
        base_time = np.array([row["base_time"] for row in rows]) * 1000
        kd_time = np.array([row["kd_time"] for row in rows]) * 1000
        calculations_fraction = np.array([
            row["kd_calculations"] / row["base_calculations"] for row in rows
        ])
        speedup = base_time / kd_time
        row_axes = axes[row_index]

        row_axes[0].plot(
            x,
            base_time,
            "o-",
            color=BASE_COLOR,
            label="Полный перебор",
        )
        row_axes[0].plot(
            x,
            kd_time,
            "o-",
            color=KD_COLOR,
            label="KD-дерево",
        )
        row_axes[0].set(
            xlabel=xlabel,
            ylabel=f"{distribution}\nмс на запрос",
            title="Время поиска",
        )
        row_axes[0].legend()

        row_axes[1].plot(x, calculations_fraction, "o-", color=KD_COLOR)
        row_axes[1].set(
            xlabel=xlabel,
            ylabel="доля от полного перебора",
            title="Вычисленные расстояния",
        )

        row_axes[2].plot(x, speedup, "o-", color=KD_COLOR)
        row_axes[2].axhline(1, color=BASE_COLOR, linestyle="--")
        row_axes[2].set(
            xlabel=xlabel,
            ylabel="время перебора / время KD",
            title="Ускорение",
        )

        for axis in row_axes:
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


def plot_kernel_benchmark(results, title, filename):
    distributions = list(dict.fromkeys(row["distribution"] for row in results))
    kernels = list(dict.fromkeys(row["kernel"] for row in results))
    colors = dict(zip(kernels, KERNEL_COLORS))

    figure, axes = plt.subplots(
        len(distributions),
        3,
        figsize=(16, 4 * len(distributions)),
        squeeze=False,
        layout="constrained",
    )
    figure.suptitle(title, fontsize=16)

    for row_index, distribution in enumerate(distributions):
        distribution_rows = [
            row for row in results
            if row["distribution"] == distribution
        ]
        row_axes = axes[row_index]

        for kernel in kernels:
            rows = [row for row in distribution_rows if row["kernel"] == kernel]
            h = np.array([row["h"] for row in rows])
            base_time = np.array([row["base_time"] for row in rows]) * 1000
            kd_time = np.array([row["kd_time"] for row in rows]) * 1000
            kd_calculations = np.array([
                row["kd_calculations"] for row in rows
            ])
            speedup = base_time / kd_time
            color = colors[kernel]

            row_axes[0].plot(
                h,
                base_time,
                "o--",
                color=color,
                alpha=0.6,
                label=f"{kernel}: перебор",
            )
            row_axes[0].plot(
                h,
                kd_time,
                "o-",
                color=color,
                label=f"{kernel}: KD",
            )
            row_axes[1].plot(
                h,
                kd_calculations,
                "o-",
                color=color,
                label=kernel,
            )
            row_axes[2].plot(
                h,
                speedup,
                "o-",
                color=color,
                label=kernel,
            )

        base_calculations = distribution_rows[0]["base_calculations"]
        row_axes[1].axhline(
            base_calculations,
            color="#666666",
            linestyle="--",
            label="Полный перебор",
        )
        row_axes[2].axhline(1, color="#666666", linestyle="--")

        row_axes[0].set(
            xlabel="Ширина окна h",
            ylabel=f"{distribution}\nмс на запрос",
            title="Время поиска",
        )
        row_axes[1].set(
            xlabel="Ширина окна h",
            ylabel="количество точек",
            title="Вычисленные расстояния KD-дерева",
        )
        row_axes[2].set(
            xlabel="Ширина окна h",
            ylabel="время перебора / время KD",
            title="Ускорение",
        )

        row_axes[0].legend(fontsize=8, ncol=2)
        row_axes[1].legend(fontsize=8)
        row_axes[2].legend(fontsize=8)

        for axis in row_axes:
            axis.set_xscale("log")
            axis.grid(alpha=0.3)

    output = Path(__file__).resolve().parents[1] / "images" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    return output
