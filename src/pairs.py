"""Finding undercut attempts, and finding something to compare them with.

An **undercut** is a car pitting earlier than the car it is chasing, so
that it spends a few laps on fresh tyres while the rival is still on old
ones, and hopes to be ahead when the rival finally stops.

The question this project asks is what that is worth, which is harder than
it sounds, because teams do not call an undercut at random. They call one
when they think it will work. So the raw success rate of attempted
undercuts is not the effect of undercutting; it is the effect of
undercutting *plus* the judgement of the people who chose to try. The same
difficulty as the tyre cliff in the strategy project, and it needs the same
treatment: something to compare against.

**The control.** Pairs of cars that pitted on the *same* lap. They are
subject to every other thing that happens in a race window - fuel burn,
track evolution, traffic, one car simply being quicker - and to no
undercut at all, because neither has a tyre advantage over the other at
any point. Their gap change over the same window is what the measurement
should read when the undercut is worth nothing.

**The outcome.** Everything is anchored on the moment before the first of
the two stops, and read at horizons counted from the moment the *second*
car has stopped:

    delta(h)  =  gap(before the first stop) - gap(h laps after the second)

Positive means the chasing car closed. Measuring from after the second
stop matters: before it, one car has paid a twenty-second pit loss and the
other has not, so the gap is dominated by a pit loss that is about to
cancel and carries no information about the undercut.

**What is excluded, and why.** Every rule is structural and declared in
advance.

`neutralised`   any window containing a safety car lap. Field spread
                collapses and pit losses change by two thirds, so the
                arithmetic the window is measuring does not hold.
`conditions`    races whose own pace drifts too far, by the measure
                carried over from the degradation project.
`too far apart` pairs more than `MAX_GAP` seconds apart before the stop.
                An undercut is not being attempted across that much, and
                the filter also removes lapped cars, whose gap carries a
                whole lap.
`extra stops`   either car stopping again inside the window. The second
                stop's pit loss swamps the thing being measured.
`unfinished`    either car not completing the window.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations

import numpy as np

import gaps
import stints
from data import Lap, Result, Stop

MAX_GAP = 25.0        # seconds; beyond this nobody is attempting anything
MAX_ADVANTAGE = 6     # laps of head start still called an undercut
HORIZONS = (0, 1, 2, 3, 5, 8, 12)
PACE_WINDOW = 8       # laps before the stop used to measure prior pace


@dataclass(frozen=True)
class Attempt:
    """One ordered pair of cars across one stop window."""

    season: int
    round: int
    circuit: str
    ahead: str            # the car being chased
    behind: str           # the car doing the undercutting
    first_stop: int       # lap the chasing car pitted on
    advantage: int        # laps of head start, 0 for a control pair
    rival_age: int        # the chased car's tyre age at that moment
    own_age: int          # the chasing car's tyre age at that moment
    gap_before: float
    prior_pace: float     # s/lap the chaser was losing before the window
    deltas: tuple[float, ...]   # gap closed at each horizon, NaN if unusable

    @property
    def control(self) -> bool:
        return self.advantage == 0


def _stop_table(stops: list[Stop]) -> dict[gaps.Key, list[int]]:
    table: dict[gaps.Key, list[int]] = defaultdict(list)
    for stop in stops:
        table[(stop.season, stop.round, stop.driver)].append(stop.lap)
    return {key: sorted(value) for key, value in table.items()}


def build(laps: list[Lap], stops: list[Stop], results: list[Result], *,
          max_gap: float = MAX_GAP, max_advantage: int = MAX_ADVANTAGE,
          horizons: tuple[int, ...] = HORIZONS
          ) -> tuple[list[Attempt], dict[str, int]]:
    """Every measurable pair in the season, with a census of the rest."""
    times = gaps.elapsed(laps)
    stop_table = _stop_table(stops)
    circuits = {(row.season, row.round): row.circuit for row in results}

    staged, _ = stints.build(laps, stops, results, condition_spread_max=1e9)
    spreads = stints.condition_spread(staged)
    excluded = {race for race, value in spreads.items()
                if value > stints.CONDITION_SPREAD_MAX}
    neutral = stints.neutralised_laps(laps)

    cars_by_race: dict[tuple[int, int], list[gaps.Key]] = defaultdict(list)
    for key in times:
        cars_by_race[(key[0], key[1])].append(key)

    census: dict[str, int] = defaultdict(int)
    out: list[Attempt] = []
    horizon_max = max(horizons)

    for race in sorted(cars_by_race):
        if race in excluded:
            census["races dropped: changing conditions"] += 1
            continue
        flagged = neutral.get(race, set())
        cars = sorted(cars_by_race[race])

        for ahead, behind in permutations(cars, 2):
            for first_stop in stop_table.get(behind, []):
                census["candidate windows"] += 1
                before = first_stop - 1
                opening = gaps.gap(times, ahead, behind, before)
                if not np.isfinite(opening):
                    census["dropped: no gap before the stop"] += 1
                    continue
                if not 0.0 < opening <= max_gap:
                    census["dropped: not within reach"] += 1
                    continue

                later = [lap for lap in stop_table.get(ahead, [])
                         if lap >= first_stop]
                if not later:
                    census["dropped: the chased car never stopped again"] += 1
                    continue
                advantage = later[0] - first_stop
                if advantage > max_advantage:
                    census["dropped: head start too long to be an "
                           "undercut"] += 1
                    continue

                second_stop = later[0]
                window_end = second_stop + horizon_max
                if any(lap in flagged
                       for lap in range(before, window_end + 1)):
                    census["dropped: neutralised inside the window"] += 1
                    continue

                extra = [lap for key in (ahead, behind)
                         for lap in stop_table.get(key, [])
                         if second_stop < lap <= window_end]
                if extra:
                    census["dropped: another stop inside the window"] += 1
                    continue

                ages = _ages(ahead, behind, first_stop, stop_table)
                prior = gaps.pace_difference(times, ahead, behind,
                                             max(before - PACE_WINDOW, 0),
                                             before)

                deltas = []
                for horizon in horizons:
                    after = gaps.gap(times, ahead, behind,
                                     second_stop + horizon)
                    deltas.append(opening - after
                                  if np.isfinite(after) else np.nan)
                if not np.isfinite(deltas[0]):
                    census["dropped: window not completed"] += 1
                    continue

                census["kept: control (same lap)" if advantage == 0
                       else "kept: undercut attempt"] += 1
                out.append(Attempt(
                    season=race[0], round=race[1],
                    circuit=circuits.get(race, "?"),
                    ahead=ahead[2], behind=behind[2],
                    first_stop=first_stop, advantage=advantage,
                    rival_age=ages[0], own_age=ages[1],
                    gap_before=opening,
                    prior_pace=prior if np.isfinite(prior) else 0.0,
                    deltas=tuple(deltas)))
    return out, dict(census)


def _ages(ahead: gaps.Key, behind: gaps.Key, lap: int,
          stop_table: dict[gaps.Key, list[int]]) -> tuple[int, int]:
    """Tyre age of each car on the lap the first of them pitted.

    The chased car's age is the quantity the whole prediction turns on:
    the undercut is worth the pace difference between a fresh set and
    whatever the rival is on, so an old set on the car ahead is what makes
    the move work.
    """
    return (stints.assign_stint(lap, stop_table.get(ahead, []))[1],
            stints.assign_stint(lap, stop_table.get(behind, []))[1])


def arrays(attempts: list[Attempt], horizon_index: int) -> dict:
    """Columns for a regression at one horizon, clustered on the car-race
    of the chasing car - the unit a single strategy call acts on, and the
    level at which these observations are plainly not independent: one
    stop puts the same car into a pair with every rival near it."""
    usable = [row for row in attempts
              if np.isfinite(row.deltas[horizon_index])]
    labels = sorted({(row.season, row.round, row.behind) for row in usable})
    index = {label: position for position, label in enumerate(labels)}
    return {
        "delta": np.array([row.deltas[horizon_index] for row in usable]),
        "advantage": np.array([float(row.advantage) for row in usable]),
        "rival_age": np.array([float(row.rival_age) for row in usable]),
        "own_age": np.array([float(row.own_age) for row in usable]),
        "gap_before": np.array([row.gap_before for row in usable]),
        "prior_pace": np.array([row.prior_pace for row in usable]),
        "group": np.array([index[(row.season, row.round, row.behind)]
                           for row in usable]),
        "n_groups": len(labels),
        "rows": usable,
    }
