"""What the undercut is worth, and what that says about the tyres.

Two questions, and the second is the interesting one.

**How much does pitting first gain?** The chasing car's gap change is
compared with what the same pair of cars does when they stop on the *same*
lap. That control is not a formality: the car behind is behind for a
reason, and in 2023 a chasing car in one of these windows loses about two
seconds to the car ahead even when neither of them gains a tyre advantage.
Measured against zero, the undercut would look worthless. Measured against
the control, it is worth several seconds.

**What does the size of the gain say about tyre degradation?** This is the
part that reaches back into the strategy project. On the laps where one
car has fresh tyres and the other does not, the pace difference between
them *is* the tyre-age penalty, and it is being read off the gap between
two cars rather than off the slope of one car's lap times. Writing `d` for
the penalty per lap of age, `T` for the chased car's tyre age and `k` for
the laps of head start, the pure-pace prediction for the gap closed is

    delta  =  d * T * k      at the moment the chased car rejoins
              and then decays by d * k for every lap afterwards

so regressing the measured gap change on `T * k` estimates `d` directly.
That estimator is not the same one as a regression of lap time on tyre
age, and it is not selected the same way: it uses the *contrast* between
two cars inside a few laps, not the slope of a long stint that only exists
because the tyre held on. The strategy project predicted that the lap-time
estimate is biased toward zero and could not measure by how much. This can.

**Standard errors.** One pit stop puts the same car into a pair with every
rival within reach, so these observations are emphatically not
independent. Errors are clustered on the car-race of the chasing car. A
second clustering, on the race, is reported alongside: it is more
conservative and has only twenty clusters, so it is a check rather than
the headline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Fit:
    """Plain least squares with cluster-robust errors.

    Deliberately not the within estimator carried over from the earlier
    projects: that one absorbs fixed effects, and here the intercept is
    the quantity of interest - it is the control group's mean.
    """

    names: tuple[str, ...]
    beta: np.ndarray
    covariance: np.ndarray
    n: int
    groups: int
    sigma: float
    r_squared: float
    label: str = ""

    def __getitem__(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    def error(self, name: str) -> float:
        index = self.names.index(name)
        return float(np.sqrt(self.covariance[index, index]))

    def t_value(self, name: str) -> float:
        return self[name] / self.error(name)

    def p_value(self, name: str) -> float:
        return float(2 * stats.t.sf(abs(self.t_value(name)),
                                    max(self.groups - 1, 1)))

    def interval(self, name: str, alpha: float = 0.05) -> tuple[float, float]:
        half = stats.t.ppf(1 - alpha / 2, max(self.groups - 1, 1))
        spread = half * self.error(name)
        return self[name] - spread, self[name] + spread


def ols(y: np.ndarray, matrix: np.ndarray, names: tuple[str, ...],
        group: np.ndarray, *, label: str = "") -> Fit:
    """Least squares with Liang-Zeger cluster-robust covariance.

    The small-sample correction uses the number of clusters, not the
    number of rows, because the rows inside a cluster carry nothing like
    a cluster's worth of independent information.
    """
    if matrix.shape[0] != y.size:
        raise ValueError("design and outcome have different lengths")
    rank = np.linalg.matrix_rank(matrix)
    if rank < matrix.shape[1]:
        raise ValueError(
            f"the design is rank deficient (rank {rank} < "
            f"{matrix.shape[1]} columns: {', '.join(names)})")

    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ beta
    bread = np.linalg.inv(matrix.T @ matrix)

    labels = np.unique(group)
    meat = np.zeros((matrix.shape[1], matrix.shape[1]))
    for one in labels:
        rows = group == one
        score = matrix[rows].T @ residual[rows]
        meat += np.outer(score, score)

    clusters = labels.size
    correction = (clusters / max(clusters - 1, 1)) * \
                 ((y.size - 1) / max(y.size - matrix.shape[1], 1))
    covariance = bread @ meat @ bread * correction

    centred = y - y.mean()
    spread = float(centred @ centred)
    # A constant outcome has no variance to explain, so R squared is not
    # defined rather than zero. It happens on purpose in the tests, where
    # a race is built with no degradation and every pair returns the same
    # number.
    explained = (1 - float(residual @ residual) / spread
                 if spread > 0 else float("nan"))
    return Fit(names=tuple(names), beta=beta, covariance=covariance,
               n=int(y.size), groups=int(clusters),
               sigma=float(np.sqrt(residual @ residual
                                   / max(y.size - matrix.shape[1], 1))),
               r_squared=explained, label=label)


def _drift(arrays: dict, horizon: int) -> np.ndarray:
    """What each pair would have done over the window without any stop.

    The window runs from the lap before the first stop to `horizon` laps
    after the second, which is `advantage + horizon + 1` laps. A pair
    already closing at a tenth a lap closes more of it over a longer
    window, and windows are longer for larger head starts, so leaving
    this out would load a pure artefact of window length onto the thing
    being measured.
    """
    return arrays["prior_pace"] * (arrays["advantage"] + horizon + 1.0)


def gain(arrays: dict, horizon: int, *, drift: bool = True) -> Fit:
    """Gap closed, against head start and against head start times age.

    The intercept is the control group: pairs that stopped on the same
    lap, where no tyre advantage exists at any point.
    """
    n = arrays["delta"].size
    columns = [np.ones(n), arrays["advantage"],
               arrays["advantage"] * arrays["rival_age"]]
    names = ["control", "per lap of head start", "per lap of head start x age"]
    if drift:
        columns.append(_drift(arrays, horizon))
        names.append("prior pace")
    return ols(arrays["delta"], np.column_stack(columns), tuple(names),
               arrays["group"], label=f"gap closed {horizon} laps after")


def degradation_from_gaps(arrays: dict, horizon: int, *,
                          drift: bool = True) -> Fit:
    """The tyre-age penalty implied by the undercut, with no `k` term.

    Dropping the plain head-start column forces the whole gain through
    the `T * k` channel, so the coefficient is an estimate of seconds per
    lap of tyre age - the same quantity a lap-time regression estimates,
    read off a completely different feature of the data.
    """
    n = arrays["delta"].size
    columns = [np.ones(n), arrays["advantage"] * arrays["rival_age"]]
    names = ["control", "seconds per lap of age"]
    if drift:
        columns.append(_drift(arrays, horizon))
        names.append("prior pace")
    return ols(arrays["delta"], np.column_stack(columns), tuple(names),
               arrays["group"], label=f"implied degradation at h={horizon}")


def by_advantage(attempts, horizons: tuple[int, ...], horizon: int
                 ) -> list[dict]:
    """Raw means by head start, before any model touches them."""
    index = horizons.index(horizon)
    control = [row.deltas[index] for row in attempts
               if row.control and np.isfinite(row.deltas[index])]
    base = float(np.mean(control)) if control else float("nan")
    out = []
    for advantage in sorted({row.advantage for row in attempts}):
        values = [row.deltas[index] for row in attempts
                  if row.advantage == advantage
                  and np.isfinite(row.deltas[index])]
        if len(values) < 10:
            continue
        values = np.array(values)
        out.append({
            "advantage": advantage, "n": values.size,
            "mean": float(values.mean()),
            "error": float(values.std(ddof=1) / np.sqrt(values.size)),
            "against_control": float(values.mean() - base),
            "median_age": float(np.median(
                [row.rival_age for row in attempts
                 if row.advantage == advantage])),
        })
    return out


def trajectory(attempts, horizons: tuple[int, ...]) -> list[dict]:
    """The whole shape over time, by head start.

    The undercut is not a permanent gain and the shape says so: the
    chasing car's tyres are older for the rest of the stint, so whatever
    was won is handed back at `d * k` a lap. Where the measured curve
    stops coming back down is where track position has taken over from
    pace, which is the thing a single-car model cannot represent.
    """
    out = []
    for horizon in horizons:
        index = horizons.index(horizon)
        row: dict = {"horizon": horizon}
        for advantage in sorted({a.advantage for a in attempts}):
            values = [a.deltas[index] for a in attempts
                      if a.advantage == advantage
                      and np.isfinite(a.deltas[index])]
            if len(values) >= 10:
                row[advantage] = float(np.mean(values))
        out.append(row)
    return out


def positions(attempts, horizons: tuple[int, ...], horizon: int) -> list[dict]:
    """How often the chasing car actually came out ahead, and stayed there.

    Seconds are the honest measure of pace, but the undercut is attempted
    for track position, and a gain of two seconds that does not clear the
    car ahead is worth nothing on the day.
    """
    index = horizons.index(horizon)
    late = len(horizons) - 1
    out = []
    for advantage in sorted({row.advantage for row in attempts}):
        group = [row for row in attempts if row.advantage == advantage
                 and np.isfinite(row.deltas[index])]
        if len(group) < 10:
            continue
        passed = [row for row in group
                  if row.deltas[index] > row.gap_before]
        held = [row for row in passed
                if np.isfinite(row.deltas[late])
                and row.deltas[late] > row.gap_before]
        out.append({
            "advantage": advantage, "n": len(group),
            "passed": len(passed), "share": len(passed) / len(group),
            "held": len(held),
            "held_share": len(held) / len(passed) if passed else float("nan"),
        })
    return out


def payback(attempts, horizons: tuple[int, ...], *, early: int = 1,
            late: int = 12, controls: bool = True) -> Fit:
    """Does a car that got ahead have to hand the gain back?

    The pure-pace model says yes, in full: whatever the undercut won, the
    chasing car's now-older tyres pay back at `d * k` a lap until nothing
    is left. The whole point of track position is that a car cannot
    collect a pace advantage from behind, so if the move actually cleared
    the rival, the payback should not arrive.

    The outcome is the change in gap closed between the two horizons, and
    the regressor of interest is whether the chasing car was ahead at the
    early one. **The controls are not optional.** Pairs that end up ahead
    started much closer together - about three seconds against twelve -
    and a pair that starts close is a pair of cars of similar pace, which
    would drift apart slowly whatever happened. Without the starting gap
    in the model this coefficient roughly doubles, and that difference is
    composition rather than mechanism.

    Conditioning on the outcome at the early horizon would, if anything,
    bias this toward finding mean reversion - a selected-high group
    drifting back down - which is the opposite of the effect claimed, so
    the direction of that bias is conservative here.
    """
    first, last = horizons.index(early), horizons.index(late)
    usable = [row for row in attempts if not row.control
              and np.isfinite(row.deltas[first])
              and np.isfinite(row.deltas[last])]
    if not usable:
        raise ValueError("no attempt spans both horizons")

    outcome = np.array([row.deltas[last] - row.deltas[first]
                        for row in usable])
    ahead = np.array([1.0 if row.deltas[first] > row.gap_before else 0.0
                      for row in usable])
    columns = [np.ones(len(usable)), ahead]
    names = ["payback when still behind", "ahead after the move"]
    if controls:
        columns += [np.array([row.gap_before for row in usable]),
                    np.array([row.prior_pace for row in usable]),
                    np.array([float(row.advantage) for row in usable]),
                    np.array([float(row.rival_age) for row in usable])]
        names += ["gap before", "prior pace", "head start", "rival age"]

    labels = sorted({(row.season, row.round, row.behind) for row in usable})
    index = {label: position for position, label in enumerate(labels)}
    group = np.array([index[(row.season, row.round, row.behind)]
                      for row in usable])
    return ols(outcome, np.column_stack(columns), tuple(names), group,
               label=f"payback from h={early} to h={late}")


def split_trajectory(attempts, horizons: tuple[int, ...], *, early: int = 1
                     ) -> list[dict]:
    """Mean gap closed at each horizon, split by whether the move worked.

    The levels here are selected - a pair is in the `ahead` group because
    of its value at the early horizon - so the levels are not a causal
    quantity and are not read as one. The *shape* is what matters, and
    `payback` is where it is tested with controls.
    """
    first = horizons.index(early)
    usable = [row for row in attempts if not row.control
              and np.isfinite(row.deltas[first])]
    ahead = [row for row in usable if row.deltas[first] > row.gap_before]
    behind = [row for row in usable if row.deltas[first] <= row.gap_before]

    out = []
    for horizon in horizons:
        index = horizons.index(horizon)
        row: dict = {"horizon": horizon}
        for name, group in (("ahead", ahead), ("behind", behind)):
            values = [item.deltas[index] for item in group
                      if np.isfinite(item.deltas[index])]
            if values:
                row[name] = float(np.mean(values))
                row[f"{name}_n"] = len(values)
        out.append(row)
    return out
