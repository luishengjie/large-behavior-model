"""Reusable plotting utilities for exploratory data analysis."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def categorical_summary(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Calculate the count and percentage of each value in a column."""

    if column not in data.columns:
        raise KeyError(f"Column not found: {column!r}.")
    if data.empty:
        raise ValueError("Cannot summarize an empty DataFrame.")

    counts = data[column].value_counts(dropna=False)

    return pd.DataFrame(
        {
            "count": counts,
            "percentage": counts / len(data) * 100,
        }
    )


def plot_categorical_distribution(
    data: pd.DataFrame,
    column: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Number of participants",
    category_order: list[str] | None = None,
    top_n: int | None = None,
    figsize: tuple[float, float] = (8, 4),
    rotation: int = 0,
    headroom: float = 0.18,
    max_label_length: int | None = None,
) -> Axes:
    """Plot the count and percentage of each value in a categorical column."""

    if headroom < 0:
        raise ValueError("headroom must be non-negative.")
    if max_label_length is not None and max_label_length <= 3:
        raise ValueError("max_label_length must be greater than three.")

    summary = categorical_summary(data, column)

    if category_order is not None:
        summary = summary.reindex(category_order)

    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero.")
        summary = summary.nlargest(top_n, "count")

    ax = summary["count"].plot.bar(
        edgecolor="black",
        figsize=figsize,
    )

    display_name = column.replace("_", " ").title()
    ax.set_xlabel(xlabel or display_name)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{display_name} distribution", pad=12)
    ax.tick_params(axis="x", rotation=rotation)

    if max_label_length is not None:
        labels = []
        for value in summary.index:
            label = str(value)
            if len(label) > max_label_length:
                label = f"{label[: max_label_length - 3].rstrip()}..."
            labels.append(label)
        ax.set_xticklabels(labels)

    maximum_count = summary["count"].max()
    if pd.notna(maximum_count) and maximum_count > 0:
        ax.set_ylim(0, maximum_count * (1 + headroom))

    for position, (_, row) in enumerate(summary.iterrows()):
        if pd.isna(row["count"]):
            continue

        ax.annotate(
            f"{int(row['count']):,}\n({row['percentage']:.1f}%)",
            xy=(position, row["count"]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return ax
