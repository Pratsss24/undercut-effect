"""The gap between two cars, to the thousandth, with no interpolation.

A lap time feed looks like it has no positional information in it. It does.
Summing a car's lap times gives its elapsed race time at the end of every
lap, and the difference between two cars' elapsed times *is* the gap
between them at that moment:

    gap(A, B, n)  =  elapsed_B(n) - elapsed_A(n)

Positive means B is behind A. The timing line is the same line for both
cars, so this is exact rather than an estimate, and it needs no
interpolation, no sector data and no telemetry. Everything in this project
rests on it.

**Lapped cars.** A car a full lap down has a cumulative difference that
includes a whole lap time, so it appears as a gap of eighty to a hundred
seconds rather than as a lap. Nothing here detects that directly; the pair
filter's maximum gap removes it, because no undercut is attempted across
twenty-five seconds and certainly not across a lap. It is a filter that
happens to be load-bearing for a second reason, which is worth knowing.

**The standing start.** Lap one's recorded time runs from the start signal,
not from each car's own crossing of the line, so the grid's time spread is
inside lap one for everybody and cumulative sums stay comparable from the
first lap onwards. No correction is applied or needed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from data import Lap

Key = tuple[int, int, str]       # season, round, driver


def elapsed(laps: list[Lap]) -> dict[Key, np.ndarray]:
    """Cumulative race time per car, indexed so that `array[n]` is the
    elapsed time at the end of lap n. Index 0 is zero; laps the car never
    completed are NaN.
    """
    rows: dict[Key, list[Lap]] = defaultdict(list)
    for lap in laps:
        rows[(lap.season, lap.round, lap.driver)].append(lap)

    out: dict[Key, np.ndarray] = {}
    for key, car in rows.items():
        last = max(row.lap for row in car)
        times = np.full(last + 1, np.nan)
        times[0] = 0.0
        for row in car:
            times[row.lap] = row.seconds
        # A missing lap in the middle would silently shorten every later
        # elapsed time, so it poisons the whole car rather than one lap.
        if np.isnan(times[1:]).any():
            out[key] = np.full(last + 1, np.nan)
            out[key][0] = 0.0
            continue
        out[key] = np.cumsum(times)
    return out


def gap(times: dict[Key, np.ndarray], ahead: Key, behind: Key,
        lap: int) -> float:
    """Seconds `behind` is behind `ahead` at the end of `lap`.

    NaN when either car has not completed that lap, which the callers
    treat as "this pair cannot be measured here" rather than as zero.
    """
    first, second = times.get(ahead), times.get(behind)
    if first is None or second is None:
        return float("nan")
    if lap >= first.size or lap >= second.size or lap < 0:
        return float("nan")
    return float(second[lap] - first[lap])


def pace_difference(times: dict[Key, np.ndarray], ahead: Key, behind: Key,
                    start: int, end: int) -> float:
    """Average seconds per lap `behind` loses to `ahead` over a window.

    Used on laps *before* an undercut to establish which of the two was
    the quicker car to begin with. Two cars whose gap was already closing
    at a tenth a lap will close it further over the window whether or not
    anybody pits, and that drift has nothing to do with the undercut.
    """
    if end <= start:
        return float("nan")
    opened = gap(times, ahead, behind, end) - gap(times, ahead, behind, start)
    return opened / (end - start)
