"""Figures. Committed to the repository so the write-up renders anywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

PALETTE = {"ink": "#16181d", "muted": "#5b6472", "grid": "#e3e6ea",
           "good": "#17803d", "bad": "#b4232c", "accent": "#2563a8",
           "warm": "#d9a441", "plum": "#6b4c9a"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"], "axes.titlesize": 10,
    "axes.titleweight": "bold", "figure.facecolor": "white",
})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _grid(ax):
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def gain(rows, anchor: int, out_dir: Path) -> Path:
    """What the undercut is worth, against a same-lap control."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    labels = ["control" if row["advantage"] == 0 else str(row["advantage"])
              for row in rows]
    values = [row["mean"] for row in rows]
    errors = [1.96 * row["error"] for row in rows]
    colours = [PALETTE["muted"] if row["advantage"] == 0 else PALETTE["accent"]
               for row in rows]
    positions = np.arange(len(rows))
    axes[0].bar(positions, values, yerr=errors, color=colours, alpha=0.88,
                error_kw={"ecolor": PALETTE["ink"], "elinewidth": 1.1})
    axes[0].axhline(values[0], color=PALETTE["bad"], linestyle="--",
                    linewidth=1.2)
    axes[0].annotate("same-lap control", xy=(len(rows) - 1.4, values[0]),
                     xytext=(0, -14), textcoords="offset points",
                     fontsize=8, color=PALETTE["bad"])
    axes[0].axhline(0, color=PALETTE["muted"], linewidth=0.9)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("laps of head start")
    axes[0].set_ylabel(f"seconds closed, {anchor} lap after the rival rejoins")
    axes[0].set_title("Measured against zero the undercut looks worthless")
    _grid(axes[0])

    against = [row["against_control"] for row in rows[1:]]
    axes[1].bar(np.arange(len(against)), against, color=PALETTE["good"],
                alpha=0.88, width=0.62)
    axes[1].set_xticks(np.arange(len(against)))
    axes[1].set_xticklabels([row["advantage"] for row in rows[1:]])
    axes[1].set_xlabel("laps of head start")
    axes[1].set_ylabel("seconds gained against the control")
    axes[1].set_title("Measured against the control it is worth 2 to 4 s")
    _grid(axes[1])
    return _save(fig, out_dir, "gain")


def estimators(implied, lap_fit, out_dir: Path) -> Path:
    """Two estimates of the tyre-age penalty, from different features."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    gap_low, gap_high = implied.interval("seconds per lap of age")
    lap_low, lap_high = lap_fit.interval("age")
    entries = [
        ("gaps between two cars\n(this project)",
         implied["seconds per lap of age"], gap_low, gap_high,
         PALETTE["accent"]),
        ("lap times against tyre age\n(carried over)",
         lap_fit["age"], lap_low, lap_high, PALETTE["plum"]),
    ]
    for index, (label, value, low, high, colour) in enumerate(entries):
        ax.errorbar(value, index, xerr=[[value - low], [high - value]],
                    fmt="o", color=colour, markersize=7, capsize=4,
                    linewidth=1.6)
        ax.annotate(f"{value:+.4f}", xy=(value, index), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=colour, fontweight="bold")
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([entry[0] for entry in entries], fontsize=8.5)
    ax.set_ylim(-0.6, len(entries) - 0.4)
    ax.set_xlabel("seconds per lap of tyre age")
    ax.set_title("Independent estimates of the same quantity, and they agree")
    ax.annotate(f"ratio {implied['seconds per lap of age'] / lap_fit['age']:.2f}x"
                f" - the direction a cliff predicts,\nbut the intervals "
                f"overlap and one season cannot resolve it",
                xy=(0.02, 0.06), xycoords="axes fraction", fontsize=8,
                color=PALETTE["muted"])
    _grid(ax)
    return _save(fig, out_dir, "estimators")


def decay_chart(decay, lap_fit, out_dir: Path) -> Path:
    """The implied penalty read later and later: it goes away."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    horizons = [row["horizon"] for row in decay]
    values = [row["value"] for row in decay]
    lows = [row["value"] - row["low"] for row in decay]
    highs = [row["high"] - row["value"] for row in decay]
    ax.errorbar(horizons, values, yerr=[lows, highs], fmt="o-",
                color=PALETTE["accent"], markersize=5, capsize=3,
                linewidth=1.5)
    ax.axhline(0, color=PALETTE["ink"], linewidth=1.0)
    ax.axhline(lap_fit["age"], color=PALETTE["plum"], linestyle="--",
               linewidth=1.2)
    ax.annotate("lap-time estimate", xy=(horizons[-1], lap_fit["age"]),
                xytext=(-80, 6), textcoords="offset points", fontsize=8,
                color=PALETTE["plum"])
    ax.set_xlabel("laps after the chased car rejoins")
    ax.set_ylabel("implied seconds per lap of tyre age")
    ax.set_title("The pace advantage is a loan, and it is repaid in full")
    _grid(ax)
    return _save(fig, out_dir, "decay")


def trajectories(trajectory, split, out_dir: Path) -> Path:
    """The shape of the gap over time, overall and split by outcome."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    keys = sorted(k for k in trajectory[0] if k != "horizon")
    palette = [PALETTE["muted"], PALETTE["good"], PALETTE["accent"],
               PALETTE["warm"], PALETTE["plum"], PALETTE["bad"],
               PALETTE["ink"]]
    usable = [row for row in trajectory if row["horizon"] > 0]
    for index, key in enumerate(keys):
        axes[0].plot([row["horizon"] for row in usable],
                     [row.get(key, np.nan) for row in usable],
                     marker="o", markersize=3.5, linewidth=1.4,
                     color=palette[index % len(palette)],
                     label="control" if key == 0 else f"head start {key}")
    axes[0].axhline(0, color=PALETTE["ink"], linewidth=0.9)
    axes[0].set_xlabel("laps after the chased car rejoins")
    axes[0].set_ylabel("seconds closed")
    axes[0].set_title("Won quickly, given back slowly")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    _grid(axes[0])

    later = [row for row in split if row["horizon"] > 0]
    for name, colour, label in (("ahead", PALETTE["good"],
                                 "the move cleared the car"),
                                ("behind", PALETTE["bad"],
                                 "it did not")):
        axes[1].plot([row["horizon"] for row in later],
                     [row.get(name, np.nan) for row in later],
                     marker="o", markersize=4.5, linewidth=1.8,
                     color=colour, label=label)
    axes[1].axhline(0, color=PALETTE["ink"], linewidth=0.9)
    axes[1].set_xlabel("laps after the chased car rejoins")
    axes[1].set_ylabel("seconds closed")
    axes[1].set_title("Only the cars that got past keep what they won")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].annotate("levels here are selected;\nthe shape is what matters",
                     xy=(0.04, 0.44), xycoords="axes fraction", fontsize=8,
                     color=PALETTE["muted"])
    _grid(axes[1])
    return _save(fig, out_dir, "trajectories")


def position(place, naive, full, out_dir: Path) -> Path:
    """How often it works, and what getting ahead is worth."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    labels = ["control" if row["advantage"] == 0 else str(row["advantage"])
              for row in place]
    shares = [row["share"] for row in place]
    colours = [PALETTE["muted"] if row["advantage"] == 0 else PALETTE["accent"]
               for row in place]
    positions = np.arange(len(place))
    axes[0].bar(positions, shares, color=colours, alpha=0.88, width=0.62)
    for pos, row in zip(positions, place):
        axes[0].annotate(f"{row['passed']}/{row['n']}", xy=(pos, row["share"]),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=7.5, color=PALETTE["muted"])
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("laps of head start")
    axes[0].set_ylabel("share of attempts that cleared the car")
    axes[0].set_title("Three times the control's odds, and it still "
                      "usually fails")
    _grid(axes[0])

    naive_value = naive["ahead after the move"]
    full_value = full["ahead after the move"]
    naive_low, naive_high = naive.interval("ahead after the move")
    full_low, full_high = full.interval("ahead after the move")
    axes[1].barh([0, 1], [naive_value, full_value],
                 xerr=[[naive_value - naive_low, full_value - full_low],
                       [naive_high - naive_value, full_high - full_value]],
                 color=[PALETTE["bad"], PALETTE["good"]], alpha=0.88,
                 height=0.5,
                 error_kw={"ecolor": PALETTE["ink"], "elinewidth": 1.2})
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["no controls", "starting gap, pace,\nhead "
                             "start, rival age"], fontsize=8.5)
    axes[1].set_xlabel("seconds of payback avoided by getting ahead")
    axes[1].set_title("Half the raw effect was composition")
    axes[1].annotate("pairs that get ahead started\n~3 s apart; the rest "
                     "~12 s", xy=(0.04, 0.12), xycoords="axes fraction",
                     fontsize=8, color=PALETTE["muted"])
    _grid(axes[1])
    return _save(fig, out_dir, "position")


def census_chart(census, out_dir: Path) -> Path:
    """Which windows the measurement is built from."""
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    items = [(key, value) for key, value in census.items()
             if key != "candidate windows" and value]
    items.sort(key=lambda item: -item[1])
    names = [key for key, _ in items]
    values = [value for _, value in items]
    colours = [PALETTE["good"] if key.startswith("kept") else PALETTE["muted"]
               for key in names]
    positions = np.arange(len(names))
    ax.barh(positions, values, color=colours, alpha=0.88)
    for pos, value in zip(positions, values):
        ax.annotate(f"{value:,}", xy=(value, pos), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=PALETTE["muted"])
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"windows (of {census.get('candidate windows', 0):,} "
                  f"candidates)")
    ax.set_title("What is measured and what is thrown away")
    _grid(ax)
    return _save(fig, out_dir, "census")


def draw(out_dir: Path, **kwargs) -> dict[str, Path]:
    return {
        "census": census_chart(kwargs["census"], out_dir),
        "gain": gain(kwargs["rows"], kwargs["anchor"], out_dir),
        "estimators": estimators(kwargs["implied"], kwargs["lap_fit"],
                                 out_dir),
        "decay": decay_chart(kwargs["decay"], kwargs["lap_fit"], out_dir),
        "trajectories": trajectories(kwargs["trajectory"], kwargs["split"],
                                     out_dir),
        "position": position(kwargs["place"], kwargs["naive"],
                             kwargs["full"], out_dir),
    }
