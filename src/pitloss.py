"""How much a pit stop actually costs, measured rather than looked up.

The feed records a `duration` for every stop, but in this schema that field
is the **pit-lane transit time** — entry line to exit line, 18 to 30
seconds depending on the circuit. It is not the pit loss. Some of that time
would have been spent driving that stretch of track anyway, and the loss is
only the difference.

So the cost is measured from the lap times instead, which needs no
convention at all:

    loss = (in-lap  - what that lap would have taken clean)
         + (out-lap - what that lap would have taken clean)

The counterfactual comes from the car's own race. For each car-race the
clean laps are regressed on tyre age and race lap, which is the same model
the degradation project fits, and the fitted line is evaluated at the age
and lap of the in-lap and the out-lap. A car that stopped has at least two
stints, so age and race lap are separately identified and the prediction is
not an extrapolation along a collinear design.

Two numbers come out of this, not one. A stop taken while the field is
neutralised costs less, because the lap the car gives up is a slow lap, and
that discount is the single most important parameter in any strategy model
that takes safety cars seriously. It is measured here from the stops that
happened to fall in neutralised laps, by exactly the same method, with the
counterfactual still built from green-flag laps.

**What is excluded, and why.** A stop is dropped when the car retired
within three laps of it (the stop and the failure are the same event), when
the in-lap or out-lap falls outside the race, when another stop lands on
either of those laps (the two losses overlap and cannot be split), and when
the car-race has fewer than ten clean laps or a rank-deficient design. The
car-races excluded by the condition rule in `stints` are excluded here too:
a drying track makes the counterfactual meaningless.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import stints
from data import Lap, Result, Stop

MIN_CLEAN_LAPS = 10   # clean laps needed before a car-race's line is trusted
RETIREMENT_GAP = 3    # a stop this close to the end of a car's race is a DNF


@dataclass(frozen=True)
class Loss:
    """One measured pit stop cost."""

    circuit: str
    driver: str
    round: int
    lap: int
    neutralised: bool
    in_excess: float
    out_excess: float
    transit: float      # the feed's own pit-lane time, for cross-checking

    @property
    def total(self) -> float:
        return self.in_excess + self.out_excess


def _counterfactual(rows: list[Lap], ages: dict[int, int]
                    ) -> tuple[np.ndarray, float] | None:
    """Least squares of clean lap time on [1, age, lap] for one car-race.

    Returns the coefficients and the residual standard deviation, or None
    when the car-race cannot support the fit.
    """
    if len(rows) < MIN_CLEAN_LAPS:
        return None
    design = np.column_stack([
        np.ones(len(rows)),
        np.array([ages[row.lap] for row in rows], dtype=float),
        np.array([row.lap for row in rows], dtype=float)])
    if np.linalg.matrix_rank(design) < 3:
        return None
    target = np.array([row.seconds for row in rows])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    return beta, float(np.std(residual, ddof=3))


def neutralisation_factor(laps: list[Lap]) -> dict[tuple, float]:
    """How much slower than its own green baseline each race lap ran.

    The field median for the lap over the median of those medians across
    the race. One under green flag, and above one wherever the race was
    neutralised, by however much it was slowed.
    """
    table: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for lap in laps:
        table[(lap.season, lap.round)][lap.lap].append(lap.seconds)

    out: dict[tuple, float] = {}
    for race, laps_table in table.items():
        medians = {lap: float(np.median(times))
                   for lap, times in laps_table.items() if times}
        if not medians:
            continue
        baseline = float(np.median(list(medians.values())))
        for lap, value in medians.items():
            out[(race, lap)] = max(value / baseline, 1.0)
    return out


def measure(laps: list[Lap], stops: list[Stop], results: list[Result],
            *, ratio: float = stints.NEUTRALISED_RATIO,
            condition_cut: float = stints.CONDITION_SPREAD_MAX
            ) -> tuple[list[Loss], dict[str, int]]:
    """Every pit stop in the feed whose cost can be measured.

    The census counts what was dropped and why, so the kept sample can be
    read against the 912 stops the season contains.
    """
    circuits = {(row.season, row.round): row.circuit for row in results}
    # The condition measure is defined on racing laps, so the cleaning has
    # to run first with the condition rule itself disabled. Feeding it raw
    # laps instead inflates every race's spread with pit laps and safety
    # car laps and throws away most of the season.
    staged, _ = stints.build(laps, stops, results,
                             neutralised_ratio=ratio,
                             condition_spread_max=1e9)
    spreads = stints.condition_spread(staged)
    excluded = {race for race, spread in spreads.items()
                if spread > condition_cut}

    neutral = stints.neutralised_laps(laps, ratio=ratio)
    factors = neutralisation_factor(laps)
    last_lap = {(row.season, row.round, row.driver): row.laps
                for row in results}
    finished = {(row.season, row.round, row.driver):
                row.status in stints.FINISHED for row in results}

    by_car: dict[tuple[int, int, str], list[Lap]] = defaultdict(list)
    for row in laps:
        by_car[(row.season, row.round, row.driver)].append(row)
    stops_by_car: dict[tuple[int, int, str], list[Stop]] = defaultdict(list)
    for stop in stops:
        stops_by_car[(stop.season, stop.round, stop.driver)].append(stop)

    census = defaultdict(int)
    out: list[Loss] = []

    for key, car_laps in sorted(by_car.items()):
        season, round_, driver = key
        race = (season, round_)
        circuit = circuits.get(race, "?")
        car_stops = sorted(stops_by_car[key], key=lambda s: s.lap)
        census["stops"] += len(car_stops)
        if not car_stops:
            continue
        if race in excluded:
            census["dropped: changing conditions"] += len(car_stops)
            continue

        stop_laps = [stop.lap for stop in car_stops]
        times = {row.lap: row.seconds for row in car_laps}
        ages = {row.lap: stints.assign_stint(row.lap, stop_laps)[1]
                for row in car_laps}
        end = last_lap.get(key, max(times))

        # The clean laps that build the counterfactual, by the same rules
        # the degradation project uses, minus the stint-length rule, which
        # is about estimating a slope rather than predicting one lap.
        clean = [row for row in car_laps
                 if row.lap != 1
                 and row.lap not in neutral.get(race, set())
                 and row.lap not in stop_laps                 # in-laps
                 and (row.lap - 1) not in stop_laps           # out-laps
                 and not (not finished.get(key, False) and row.lap == end)]
        fitted = _counterfactual(clean, ages)
        if fitted is None:
            census["dropped: car-race not fittable"] += len(car_stops)
            continue
        beta, _ = fitted

        for stop in car_stops:
            in_lap, out_lap = stop.lap, stop.lap + 1
            if in_lap not in times or out_lap not in times:
                census["dropped: lap outside the race"] += 1
                continue
            if end - in_lap < RETIREMENT_GAP and not finished.get(key, False):
                census["dropped: retired just after"] += 1
                continue
            if any(other != stop.lap and other in (in_lap - 1, in_lap,
                                                   out_lap, out_lap + 1)
                   for other in stop_laps):
                census["dropped: overlapping stop"] += 1
                continue

            flagged = neutral.get(race, set())
            was_neutral = in_lap in flagged or out_lap in flagged
            # Under a safety car the car that stays out is slow too, so the
            # reference has to be slow as well. The car's own clean line is
            # scaled by how far the field median for that lap sits above
            # the race's green baseline. Measured against a green-flag lap
            # instead, a safety car stop comes out looking twice as
            # expensive as a normal one, which is backwards.
            predicted = [float(beta @ [1.0, ages[lap], lap])
                         * factors.get((race, lap), 1.0)
                         for lap in (in_lap, out_lap)]
            census["kept: neutralised" if was_neutral
                   else "kept: green flag"] += 1
            out.append(Loss(
                circuit=circuit, driver=driver, round=round_, lap=in_lap,
                neutralised=was_neutral,
                in_excess=times[in_lap] - predicted[0],
                out_excess=times[out_lap] - predicted[1],
                transit=stop.duration))

    return out, dict(census)


def summarise(losses: list[Loss]) -> dict:
    """Green-flag and neutralised pit loss, and the discount between them.

    Medians, not means: a stop that went wrong adds ten seconds to the
    right tail and nothing to the left, and the parameter wanted here is
    the cost of a normal stop.
    """
    green = np.array([loss.total for loss in losses if not loss.neutralised])
    neutral = np.array([loss.total for loss in losses if loss.neutralised])
    transit = np.array([loss.transit for loss in losses
                        if not loss.neutralised and np.isfinite(loss.transit)])
    return {
        "green_n": int(green.size),
        "green": float(np.median(green)),
        "green_p25": float(np.percentile(green, 25)),
        "green_p75": float(np.percentile(green, 75)),
        "green_sd": float(np.std(green)),
        "neutral_n": int(neutral.size),
        "neutral": float(np.median(neutral)) if neutral.size else float("nan"),
        "discount": (float(np.median(neutral) / np.median(green))
                     if neutral.size else float("nan")),
        "transit": float(np.median(transit)),
        "transit_gap": float(np.median(transit) - np.median(green)),
    }


def by_circuit(losses: list[Loss], *, minimum: int = 8) -> list[dict]:
    """Green-flag pit loss per circuit, where there are enough stops."""
    grouped: dict[str, list[float]] = defaultdict(list)
    transits: dict[str, list[float]] = defaultdict(list)
    for loss in losses:
        if loss.neutralised:
            continue
        grouped[loss.circuit].append(loss.total)
        if np.isfinite(loss.transit):
            transits[loss.circuit].append(loss.transit)
    rows = []
    for circuit, values in grouped.items():
        if len(values) < minimum:
            continue
        rows.append({
            "circuit": circuit,
            "loss": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "n": len(values),
            "transit": (float(np.median(transits[circuit]))
                        if transits[circuit] else float("nan")),
        })
    return sorted(rows, key=lambda row: -row["loss"])


def pace_noise(laps: list[Lap], stops: list[Stop], results: list[Result]
               ) -> float:
    """Lap-to-lap scatter of a single car around its own fitted line.

    A by-product of the same per-car-race regressions the pit loss
    measurement needs. This is the right scale for the simulation's noise
    term: the pooled residual from a fit with car-race fixed effects is
    larger, because it still carries differences between cars within a
    race, and a single-car simulation should not inherit those.
    """
    neutral = stints.neutralised_laps(laps)
    stop_table: dict[tuple, list[int]] = defaultdict(list)
    for stop in stops:
        stop_table[(stop.season, stop.round, stop.driver)].append(stop.lap)
    finished = {(r.season, r.round, r.driver): r.status in stints.FINISHED
                for r in results}
    last = {(r.season, r.round, r.driver): r.laps for r in results}

    by_car: dict[tuple, list[Lap]] = defaultdict(list)
    for row in laps:
        by_car[(row.season, row.round, row.driver)].append(row)

    scatter = []
    for key, car_laps in by_car.items():
        race = (key[0], key[1])
        stop_laps = sorted(stop_table[key])
        if len(stop_laps) < 1:
            continue
        ages = {row.lap: stints.assign_stint(row.lap, stop_laps)[1]
                for row in car_laps}
        end = last.get(key, 0)
        clean = [row for row in car_laps
                 if row.lap != 1
                 and row.lap not in neutral.get(race, set())
                 and row.lap not in stop_laps
                 and (row.lap - 1) not in stop_laps
                 and not (not finished.get(key, False) and row.lap == end)]
        fitted = _counterfactual(clean, ages)
        if fitted is not None:
            scatter.append(fitted[1])
    return float(np.median(scatter))
