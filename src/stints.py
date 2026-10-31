"""Turn a feed of lap times into stints, and decide which laps are usable.

A **stint** is the run between two pit stops. The feed gives the lap each
stop was made on, so the boundaries are known exactly even though the tyre
compound is not.

    stop on lap 18  ->  stint 1 is laps 1-18, stint 2 starts on lap 19

**Tyre age** is counted within the stint, starting at 1 for the first full
lap on the set. Age is the regressor the whole analysis turns on, and the
reason it can be separated from fuel load at all: race lap only ever goes
up, while age resets at every stop. A driver who never stops contributes
nothing to that separation — for them age and race lap are the same column
twice, and the design matrix knows it.

**Which laps are thrown away, and why.** Every rule here is structural: it
removes laps that are not a car running at its own pace on a clear track,
for a reason that can be stated in advance. None of them looks at how slow
the lap was relative to the model.

`in-lap`      the lap ending in the pit entry. Contains a pit lane
              transit; not a racing lap.
`out-lap`     the lap starting in the pit exit, on cold tyres from a
              standing release. Also where the age counter starts, so
              dropping it costs age 1 and begins the usable record at 2.
`lap 1`       a standing start, a full field and the first corner.
`neutralised` safety car, virtual safety car and red flag laps. Detected
              from the field rather than assumed: a lap whose median time
              across all running cars is far above the race's own clean
              baseline. The threshold is declared and its sensitivity is
              reported.
`retirement`  the last lap of a car that did not finish, which usually
              contains the failure that ended its race.
`conditions`  every lap of a race whose own pace drifts too far over its
              distance. A drying track makes the whole field faster lap
              after lap, which is a race-lap effect of a size no fuel load
              can produce, and it swamps everything the model is for. The
              measure is the spread of the field median across the racing
              laps, and the cut sits in a wide empty gap in the data
              rather than at a round number — see `condition_spread`.

What is deliberately **not** filtered is a slow lap in traffic. There is no
rule here that removes laps for being slow, because degradation makes laps
slow: a trim of the slowest laps in a stint removes the end of the stint,
which is the part carrying the signal, and biases the estimate toward zero.
Contamination of that kind is handled where it belongs, in the fit, by
Huber weights that down-weight large residuals symmetrically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from data import Lap, Result, Stop

NEUTRALISED_RATIO = 1.15     # field median above this multiple of baseline
CONDITION_SPREAD_MAX = 0.08  # how much a race's own pace may drift
MIN_STINT_LAPS = 4           # usable laps needed before a stint counts
FINISHED = ("Finished",)


@dataclass(frozen=True)
class StintLap:
    """One usable racing lap, with everything the model needs."""

    season: int
    round: int
    race: str
    circuit: str
    driver: str
    team: str
    lap: int                 # race lap, 1-based
    stint: int               # stint index, 1-based
    age: int                 # laps on this set of tyres, 1-based
    seconds: float
    position: int

    @property
    def key(self) -> tuple[int, int, str]:
        """Identifies the car-race: the unit a fixed effect is fitted to."""
        return (self.season, self.round, self.driver)


def stop_laps(stops: list[Stop]) -> dict[tuple[int, int, str], list[int]]:
    """The in-lap of every stop, per car-race, in order."""
    table: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for stop in stops:
        table[(stop.season, stop.round, stop.driver)].append(stop.lap)
    return {key: sorted(value) for key, value in table.items()}


def assign_stint(lap: int, stops: list[int]) -> tuple[int, int]:
    """(stint index, tyre age) for a race lap, given this car's stop laps.

    A stop on lap 18 ends stint 1 on lap 18 and starts stint 2 on lap 19,
    whose age is 1. The age of a lap is therefore its offset from the start
    of the stint it belongs to.
    """
    index = 0
    start = 1
    for stop in stops:
        if lap <= stop:
            break
        index += 1
        start = stop + 1
    return index + 1, lap - start + 1


def neutralised_laps(laps: list[Lap], *, ratio: float = NEUTRALISED_RATIO
                     ) -> dict[tuple[int, int], set[int]]:
    """Laps run behind a safety car, found from the field's own times.

    For each race lap, the median time across every car still running. The
    baseline is the median of those medians over the race, which is robust
    to a handful of neutralised laps however slow they are. Any lap whose
    field median exceeds `NEUTRALISED_RATIO` times the baseline is out.

    This is a measurement, not an assumption: the field slows together only
    when something has neutralised the race.
    """
    by_race: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for lap in laps:
        by_race[(lap.season, lap.round)][lap.lap].append(lap.seconds)

    out: dict[tuple[int, int], set[int]] = {}
    for race, table in by_race.items():
        medians = {lap: float(np.median(times))
                   for lap, times in table.items() if times}
        if not medians:
            out[race] = set()
            continue
        baseline = float(np.median(list(medians.values())))
        out[race] = {lap for lap, value in medians.items()
                     if value > ratio * baseline}
    return out


def condition_spread(rows: list["StintLap"]) -> dict[tuple[int, int], float]:
    """How far a race's own pace drifts, as a fraction of its own level.

    For each race lap the median time across the field, then the 5th-to-
    95th-percentile spread of those medians over the race, divided by the
    race's median. A dry race holds this inside a few per cent: the field
    gets a little quicker as fuel burns off. A race that starts or ends wet
    does not, because a drying line is worth whole seconds a lap.

    The separation is not a judgement call. In the 2023 season the measure
    runs from 0.012 to 0.050 for twenty races and then jumps to 0.124 and
    0.129 for the two that were rained on, so any cut placed in that empty
    band removes the same two races.
    """
    by_race: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        by_race[(row.season, row.round)][row.lap].append(row.seconds)

    out: dict[tuple[int, int], float] = {}
    for race, table in by_race.items():
        medians = np.array([float(np.median(v)) for v in table.values()])
        level = float(np.median(medians))
        if level <= 0 or medians.size < 5:
            out[race] = float("inf")
            continue
        out[race] = float((np.percentile(medians, 95)
                           - np.percentile(medians, 5)) / level)
    return out


def retirement_laps(results: list[Result]) -> dict[tuple[int, int, str], int]:
    """The final lap of each car that did not finish."""
    return {(r.season, r.round, r.driver): r.laps
            for r in results if r.status not in FINISHED}


def build(laps: list[Lap], stops: list[Stop], results: list[Result],
          *, neutralised_ratio: float = NEUTRALISED_RATIO,
          condition_spread_max: float = CONDITION_SPREAD_MAX,
          min_stint_laps: int = MIN_STINT_LAPS
          ) -> tuple[list[StintLap], dict[str, int]]:
    """Apply the rules above; return the usable laps and a census of losses."""
    neutral = neutralised_laps(laps, ratio=neutralised_ratio)
    stop_table = stop_laps(stops)
    retirements = retirement_laps(results)
    teams = {(r.season, r.round, r.driver): r.team for r in results}

    census = {"total": len(laps), "lap one": 0, "neutralised": 0,
              "in-lap": 0, "out-lap": 0, "retirement": 0,
              "conditions": 0, "short stint": 0, "kept": 0}

    staged: list[StintLap] = []
    for lap in laps:
        key = (lap.season, lap.round, lap.driver)
        if lap.lap == 1:
            census["lap one"] += 1
            continue
        if lap.lap in neutral.get((lap.season, lap.round), ()):
            census["neutralised"] += 1
            continue
        if retirements.get(key) == lap.lap:
            census["retirement"] += 1
            continue

        car_stops = stop_table.get(key, [])
        if lap.lap in car_stops:
            census["in-lap"] += 1
            continue
        stint, age = assign_stint(lap.lap, car_stops)
        if stint > 1 and age == 1:
            census["out-lap"] += 1
            continue

        staged.append(StintLap(
            season=lap.season, round=lap.round, race=lap.race,
            circuit=lap.circuit, driver=lap.driver,
            team=teams.get(key, "unknown"), lap=lap.lap, stint=stint,
            age=age, seconds=lap.seconds, position=lap.position))

    # Changing conditions are measured on the racing laps, which is why
    # this comes after the neutralised ones are already out: a safety car
    # would otherwise register as weather.
    spread = condition_spread(staged)
    changeable = {race for race, value in spread.items()
                  if value > condition_spread_max}
    if changeable:
        kept_staged = []
        for row in staged:
            if (row.season, row.round) in changeable:
                census["conditions"] += 1
            else:
                kept_staged.append(row)
        staged = kept_staged

    # A stint with almost no usable laps carries no slope, and a one-lap
    # stint absorbed into a car-race fixed effect contributes nothing but
    # a degree of freedom.
    counts: dict[tuple[int, int, str, int], int] = defaultdict(int)
    for row in staged:
        counts[(row.season, row.round, row.driver, row.stint)] += 1

    kept = []
    for row in staged:
        if counts[(row.season, row.round, row.driver, row.stint)] < min_stint_laps:
            census["short stint"] += 1
            continue
        kept.append(row)
    census["kept"] = len(kept)
    return kept, census


# -- the arrays the model works on ----------------------------------------

def arrays(rows: list[StintLap]) -> dict[str, np.ndarray]:
    """Columns, plus the car-race grouping the fixed effects are fitted to."""
    keys = {}
    group = np.empty(len(rows), dtype=int)
    for i, row in enumerate(rows):
        key = row.key
        if key not in keys:
            keys[key] = len(keys)
        group[i] = keys[key]
    return {
        "seconds": np.array([r.seconds for r in rows], dtype=float),
        "age": np.array([r.age for r in rows], dtype=float),
        "lap": np.array([r.lap for r in rows], dtype=float),
        "stint": np.array([r.stint for r in rows], dtype=int),
        "group": group,
        "n_groups": len(keys),
    }


def stint_count(rows: list[StintLap]) -> dict[tuple, int]:
    """Distinct stints per car-race — the identification currency."""
    table: dict[tuple, set[int]] = defaultdict(set)
    for row in rows:
        table[row.key].add(row.stint)
    return {key: len(value) for key, value in table.items()}


def identifying_share(rows: list[StintLap]) -> float:
    """Share of laps belonging to a car-race with two or more stints.

    Only those car-races can separate tyre age from race lap. A dataset
    where this is zero cannot answer the question at all, however many
    laps it has.
    """
    counts = stint_count(rows)
    if not rows:
        return 0.0
    usable = sum(1 for row in rows if counts[row.key] >= 2)
    return usable / len(rows)
