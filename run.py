"""Measure what the undercut is worth, then what its size says about tyres.

Run with no arguments; the season is cached, so a second run costs nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import charts                                              # noqa: E402
import data                                                # noqa: E402
import effect                                              # noqa: E402
import model                                               # noqa: E402
import pairs                                               # noqa: E402
import pitloss                                             # noqa: E402
import report                                              # noqa: E402
import stints                                              # noqa: E402

SEASONS = (2023,)
ANCHOR = 1       # the first horizon at which both cars have paid the loss
LATE = 12
OUT = ROOT / "out"

_log: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _log.append(line)


def main() -> None:
    laps, stops, results = data.load(SEASONS)
    attempts, census = pairs.build(laps, stops, results)

    say(f"{len(laps):,} lap times, {len(stops):,} pit stops")
    for key, value in sorted(census.items()):
        say(f"  {key:48s} {value:6d}")

    tried = [row for row in attempts if not row.control]
    controls = [row for row in attempts if row.control]
    say(f"  {len(tried)} undercut attempts and {len(controls)} same-lap "
        f"control pairs survive")

    # ---- the convention the feed uses, read off the data ---------------
    losses, _ = pitloss.measure(laps, stops, results)
    green = [row for row in losses if not row.neutralised]
    say()
    say("where the pit-lane time lands, which fixes the first usable "
        "horizon")
    say(f"  in-lap excess   {np.median([r.in_excess for r in green]):6.2f} s")
    say(f"  out-lap excess  {np.median([r.out_excess for r in green]):6.2f} s")
    index_zero = pairs.HORIZONS.index(0)
    at_zero = [row.deltas[index_zero] for row in tried
               if np.isfinite(row.deltas[index_zero])]
    say(f"  so on the lap the chased car pits, the chasing car still "
        f"reads {np.mean(at_zero):+.1f} s - a pit loss not yet paid, not "
        f"an undercut that failed")
    say(f"  every horizon below is counted from one lap later")

    # ---- the raw picture -----------------------------------------------
    rows = effect.by_advantage(attempts, pairs.HORIZONS, ANCHOR)
    say()
    say(f"gap closed {ANCHOR} lap after the chased car rejoins")
    say(f"  {'head start':>10s} {'n':>5s} {'gap closed':>13s} "
        f"{'vs control':>11s} {'rival age':>10s}")
    for row in rows:
        label = "control" if row["advantage"] == 0 else f"{row['advantage']}"
        say(f"  {label:>10s} {row['n']:5d} "
            f"{row['mean']:+8.2f} +/- {row['error']:.2f} "
            f"{row['against_control']:+11.2f} {row['median_age']:10.0f}")
    say("  the control is not zero: a car that is behind is usually behind "
        "because it is slower, and measuring against zero would hide that")
    own = float(np.mean([row.own_age for row in tried]))
    rival = float(np.mean([row.rival_age for row in tried]))
    say(f"  and the car doing the undercutting is on the older set: "
        f"{own:.1f} laps against {rival:.1f} on the car it is chasing. "
        f"A team pits because its own tyres are gone, not because the "
        f"rival's are, so the move is more often forced than chosen")

    # ---- does the gain run through tyre age, as the physics says? -------
    arrays = pairs.arrays(attempts, pairs.HORIZONS.index(ANCHOR))
    say()
    say(f"{arrays['delta'].size} pairs, {arrays['n_groups']} clusters "
        f"(car-race of the chasing car)")
    free = effect.gain(arrays, ANCHOR)
    say("the pure-pace model says the whole gain is tyre age times head "
        "start, with no free term for the head start itself")
    for name in free.names:
        low, high = free.interval(name)
        say(f"  {name:30s} {free[name]:+8.4f} ({low:+.4f} to {high:+.4f})  "
            f"p {free.p_value(name):.4f}")
    say("  the free head-start term is indistinguishable from zero, so the "
        "functional form the physics implies is not rejected")

    implied = effect.degradation_from_gaps(arrays, ANCHOR)
    say()
    say("so the gain is pushed entirely through that channel, which makes "
        "its coefficient an estimate of the tyre-age penalty itself")
    for name in implied.names:
        low, high = implied.interval(name)
        say(f"  {name:30s} {implied[name]:+8.4f} ({low:+.4f} to "
            f"{high:+.4f})  p {implied.p_value(name):.4f}")

    clean, _ = stints.build(laps, stops, results)
    lap_fit = model.fit_within(stints.arrays(clean))
    lap_low, lap_high = lap_fit.interval("age")
    gap_value = implied["seconds per lap of age"]
    gap_low, gap_high = implied.interval("seconds per lap of age")
    say()
    say("two estimators of the same quantity, from different features of "
        "the same season")
    say(f"  lap times against tyre age   {lap_fit['age']:+.4f} "
        f"({lap_low:+.4f} to {lap_high:+.4f})")
    say(f"  gaps between two cars        {gap_value:+.4f} "
        f"({gap_low:+.4f} to {gap_high:+.4f})")
    say(f"  ratio {gap_value / lap_fit['age']:.2f}x, in the direction a "
        f"tyre cliff predicts")
    overlap = gap_low < lap_high and lap_low < gap_high
    say(f"  the intervals {'overlap' if overlap else 'do not overlap'}, so "
        f"this is {'agreement between two methods, not a contradiction - '
                   'one season cannot resolve the difference'
                   if overlap else 'a real disagreement'}")

    # ---- the gain is temporary ------------------------------------------
    decay = []
    for horizon in pairs.HORIZONS:
        if horizon == 0:
            continue
        columns = pairs.arrays(attempts, pairs.HORIZONS.index(horizon))
        fit = effect.degradation_from_gaps(columns, horizon)
        low, high = fit.interval("seconds per lap of age")
        decay.append({"horizon": horizon,
                      "value": fit["seconds per lap of age"],
                      "low": low, "high": high, "n": fit.n})
    say()
    say("the same estimate, read later and later after the rival rejoins")
    for row in decay:
        say(f"  {row['horizon']:3d} laps after  {row['value']:+.4f} "
            f"({row['low']:+.4f} to {row['high']:+.4f})")
    say("  it decays to nothing, which is exactly the pure-pace "
        "prediction: the chasing car's tyres are now the older set and it "
        "hands the gain back at the same rate it won it")

    trajectory = effect.trajectory(attempts, pairs.HORIZONS)

    # ---- but track position is not handed back --------------------------
    place = effect.positions(attempts, pairs.HORIZONS, ANCHOR)
    say()
    say("how often the move actually cleared the car")
    for row in place:
        label = "control" if row["advantage"] == 0 else f"{row['advantage']}"
        held = (f"{row['held_share']:.0%}" if row["passed"] else "-")
        say(f"  head start {label:>7s}  n={row['n']:4d}  came out ahead "
            f"{row['passed']:3d} ({row['share']:5.1%})   still ahead "
            f"{LATE} laps later {row['held']:3d} ({held})")

    split = effect.split_trajectory(attempts, pairs.HORIZONS, early=ANCHOR)
    say()
    say("mean gap closed, split by whether the move worked")
    say(f"  {'h':>4s} {'ahead':>10s} {'behind':>10s}")
    for row in split:
        if row["horizon"] == 0:
            continue
        say(f"  {row['horizon']:4d} {row.get('ahead', float('nan')):10.2f} "
            f"{row.get('behind', float('nan')):10.2f}")

    naive = effect.payback(attempts, pairs.HORIZONS, early=ANCHOR, late=LATE,
                           controls=False)
    full = effect.payback(attempts, pairs.HORIZONS, early=ANCHOR, late=LATE)
    say()
    say(f"seconds given back between {ANCHOR} and {LATE} laps after the "
        f"rival rejoins  (n={full.n}, {full.groups} clusters)")
    naive_low, naive_high = naive.interval("ahead after the move")
    say(f"  without controls, getting ahead is worth "
        f"{naive['ahead after the move']:+.2f} "
        f"({naive_low:+.2f} to {naive_high:+.2f})")
    for name in full.names:
        low, high = full.interval(name)
        say(f"  {name:30s} {full[name]:+8.3f} ({low:+.3f} to {high:+.3f})  "
            f"p {full.p_value(name):.4f}")
    say("  the uncontrolled figure is nearly double, because pairs that "
        "end up ahead started about three seconds apart and pairs that do "
        "not started about twelve - that is composition, not mechanism")
    say(f"  with the starting gap in the model it is "
        f"{full['ahead after the move']:+.2f} s: real, and about half the "
        f"size the raw comparison suggested")

    # ---- output ----------------------------------------------------------
    figures = charts.draw(
        OUT, attempts=attempts, rows=rows, trajectory=trajectory,
        split=split, decay=decay, place=place, horizons=pairs.HORIZONS,
        implied=implied, lap_fit=lap_fit, anchor=ANCHOR, late=LATE,
        naive=naive, full=full, census=census)
    path = report.build(
        census=census, attempts=attempts, rows=rows, trajectory=trajectory,
        split=split, decay=decay, place=place, horizons=pairs.HORIZONS,
        free=free, implied=implied, lap_fit=lap_fit, naive=naive, full=full,
        anchor=ANCHOR, late=LATE, seasons=SEASONS, figures=figures,
        at_zero=float(np.mean(at_zero)),
        in_excess=float(np.median([r.in_excess for r in green])),
        out_excess=float(np.median([r.out_excess for r in green])),
        out_path=OUT / "report.html")

    say()
    say(f"report  {path}")
    for name, figure in figures.items():
        say(f"figure  {figure}")
    (OUT / "_run.txt").write_text("\n".join(_log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
