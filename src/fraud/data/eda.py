"""Exploratory data analysis of the raw dataset, for the slides.

Owner: Jenitha. Run with `python -m fraud.data.eda`.
Writes three PNGs and a summary JSON to reports/eda/ (git-ignored, regenerate
any time): class imbalance, Amount by class, and transaction/fraud patterns by
hour. Uses the raw data only; it never reads the held-out test set.
"""

import json
import logging

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed (also works in CI/Airflow)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraud import config  # noqa: E402
from fraud.data.ingest import load_raw, summarize  # noqa: E402

logger = logging.getLogger(__name__)

EDA_DIR = config.REPORTS_DIR / "eda"

# Colors: validated categorical slots 1-2 on a light surface (colorblind-safe
# pair). Legitimate = blue, fraud = orange, used consistently in every chart.
LEGIT_COLOR = "#2a78d6"
FRAUD_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

SECONDS_PER_HOUR = 3600


def _style_axes(ax: plt.Axes) -> None:
    """Apply the recessive chart chrome: light grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_figure(nrows: int = 1, **kwargs) -> tuple[plt.Figure, np.ndarray]:
    """Create a slide-sized figure on the chart surface color."""
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 5.6), facecolor=SURFACE, **kwargs)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style_axes(ax)
    return fig, axes


def plot_class_balance(df: pd.DataFrame, path) -> None:
    """Horizontal bar chart of legitimate vs fraud counts, labeled with count and share."""
    counts = df[config.TARGET].value_counts().reindex([0, 1], fill_value=0)
    shares = counts / counts.sum()
    fig, (ax,) = _new_figure()
    ax.barh(["Fraud", "Legitimate"], [counts[1], counts[0]],
            color=[FRAUD_COLOR, LEGIT_COLOR], height=0.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for y, cls in enumerate([1, 0]):
        # Label sits just past the bar end; the fraud bar is too short to hold text.
        ax.text(counts[cls] + counts.max() * 0.01, y,
                f"{counts[cls]:,}  ({shares[cls]:.2%})", va="center", color=INK, fontsize=11)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(f"Class imbalance: 1 fraud per {counts[0] // max(counts[1], 1):,} legitimate "
                 "transactions", loc="left", color=INK, fontsize=13)
    ax.set_xlabel("Transactions", color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_amount_by_class(df: pd.DataFrame, path) -> None:
    """Overlaid density histograms of Amount (log scale) for each class.

    Densities, not counts: with 0.17% fraud, a count histogram would make the
    fraud distribution invisible.
    """
    log_amount = np.log10(df["Amount"] + 1)  # +1 keeps the many 0.00 amounts finite
    bins = np.linspace(0, log_amount.max(), 50)
    fig, (ax,) = _new_figure()
    for cls, label, color in [(0, "Legitimate", LEGIT_COLOR), (1, "Fraud", FRAUD_COLOR)]:
        mask = df[config.TARGET] == cls
        median = df.loc[mask, "Amount"].median()
        ax.hist(log_amount[mask], bins=bins, density=True, histtype="step", linewidth=2,
                color=color, label=f"{label} (median ${median:,.2f})")
    ticks = [0, 1, 10, 100, 1_000, 10_000]
    ax.set_xticks(np.log10(np.array(ticks) + 1), [f"${t:,}" for t in ticks])
    ax.set_xlabel("Transaction amount (log scale)", color=INK_MUTED)
    ax.set_ylabel("Density", color=INK_MUTED)
    ax.set_title("Amount by class: fraud clusters at $0-$1 and around $100", loc="left",
                 color=INK, fontsize=13)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per hour of day: transaction count and fraud rate (%).

    `Time` is seconds since the first transaction in a 2-day window, so hour of
    day is (Time // 3600) mod 24, relative to the first transaction.
    """
    hour = (df["Time"] // SECONDS_PER_HOUR % 24).astype(int)
    grouped = df.groupby(hour)[config.TARGET]
    profile = pd.DataFrame({"transactions": grouped.size(), "fraud_rate_pct": grouped.mean() * 100})
    return profile.reindex(range(24), fill_value=0)


def plot_time_patterns(df: pd.DataFrame, path) -> None:
    """Two stacked panels sharing the hour axis: volume per hour, fraud rate per hour.

    Two panels instead of one dual-axis chart, because the two measures have
    different units and scales.
    """
    profile = hourly_profile(df)
    fig, (top, bottom) = _new_figure(nrows=2, sharex=True)
    top.bar(profile.index, profile["transactions"], color=LEGIT_COLOR, width=0.8)
    top.set_title("Transactions per hour of day", loc="left", color=INK, fontsize=12)
    top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bottom.plot(profile.index, profile["fraud_rate_pct"], color=FRAUD_COLOR, linewidth=2,
                marker="o", markersize=5)
    peak = profile["fraud_rate_pct"].idxmax()
    bottom.annotate(f"peak {profile.loc[peak, 'fraud_rate_pct']:.2f}% at hour {peak}",
                    (peak, profile.loc[peak, "fraud_rate_pct"]), textcoords="offset points",
                    xytext=(8, -4), color=INK, fontsize=10)
    bottom.set_title("Fraud rate per hour of day (%): highest when volume is lowest",
                     loc="left", color=INK, fontsize=12)
    bottom.set_xlabel("Hour of day (relative to first transaction)", color=INK_MUTED)
    bottom.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, out_dir=EDA_DIR) -> dict:
    """Write all EDA charts and summary.json to `out_dir`; return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_class_balance(df, out_dir / "class_balance.png")
    plot_amount_by_class(df, out_dir / "amount_by_class.png")
    plot_time_patterns(df, out_dir / "time_patterns.png")
    summary = summarize(df)
    summary["median_amount_legit"] = float(df.loc[df[config.TARGET] == 0, "Amount"].median())
    summary["median_amount_fraud"] = float(df.loc[df[config.TARGET] == 1, "Amount"].median())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Load the raw data and write the EDA outputs to reports/eda/."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_eda(load_raw())
    logger.info("EDA written to %s", EDA_DIR)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
