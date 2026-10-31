"""Separating tyre degradation from fuel burn, and what happens if you don't.

A lap time carries at least four things at once:

    seconds = car-race pace + fuel effect + track evolution
              + tyre degradation + noise

The first term is why a Red Bull lap is not a Haas lap, and why Monaco is
not Monza. It is absorbed by a **fixed effect per car-race**: one intercept
for every (season, round, driver). Nothing in the comparison then rests on
which cars happened to run which strategies, because every slope is
estimated within a car's own race.

The second and third both move with **race lap** and in the same direction:
the car gets lighter and the track rubbers in, so lap times fall. They
cannot be separated from one another by this data — both are smooth
functions of the lap counter — so they are estimated as one coefficient and
reported as one. The README says so rather than quietly calling it "fuel".

The fourth moves with **tyre age**, which is not the same column: age
resets to zero at every pit stop while the race lap counter does not. That
reset is the entire identification argument. Formally, for a car with
stints starting at laps s_1 < s_2 < ...,

    age = lap - s_k + 1    on stint k

so age and lap differ by a step function that is constant within a stint
and changes at each stop. A car with one stint has no step: its age column
is its lap column shifted, the two are collinear, and it contributes
nothing to the split. A car with two or more does, and `identifying_share`
in `stints` reports how much of the data that is.

**Why the naive estimate is biased, and in which direction.** Regress
seconds on age within stints and leave race lap out, and the fuel effect
does not vanish — it is absorbed into the age coefficient, because within a
stint age and lap move together one for one. Fuel makes laps faster as the
race goes on; degradation makes them slower. The omitted term therefore
pulls the estimate *down*, and the naive number understates degradation by
roughly the whole size of the fuel effect. Where real degradation is small,
it can come out negative: tyres apparently improving with age.

**How it is fitted.** The fixed effects are absorbed by the within
transformation — subtract each car-race's own mean from every column —
rather than by building thousands of dummy columns. By the
Frisch-Waugh-Lovell theorem the slopes are identical either way, which is
not a claim to take on trust: a test fits both on the same small sample and
requires them to agree to twelve decimals.

Standard errors are clustered by car-race. Laps within one car's race are
not independent draws — a driver nursing a car is slow for twenty laps in a
row — and ignoring that would overstate the precision of every number here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

HUBER_K = 1.345              # 95% efficiency against a normal
HUBER_ITERATIONS = 40
HUBER_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Fit:
    """A fitted linear model with cluster-robust standard errors."""

    names: tuple[str, ...]
    beta: np.ndarray
    se: np.ndarray
    n: int
    groups: int
    sigma: float
    r_squared: float
    method: str = "within, cluster-robust"
    weights: np.ndarray | None = field(default=None, repr=False)
    residuals: np.ndarray | None = field(default=None, repr=False)

    def __getitem__(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    def error(self, name: str) -> float:
        return float(self.se[self.names.index(name)])

    def t_value(self, name: str) -> float:
        return self[name] / self.error(name)

    def p_value(self, name: str) -> float:
        """Against the cluster count, not the lap count.

        With clustered errors the degrees of freedom come from the number
        of independent units, which is car-races, not laps.
        """
        return float(2.0 * stats.t.sf(abs(self.t_value(name)),
                                      max(self.groups - 1, 1)))

    def interval(self, name: str, alpha: float = 0.05) -> tuple[float, float]:
        critical = stats.t.ppf(1.0 - alpha / 2.0, max(self.groups - 1, 1))
        half = critical * self.error(name)
        return (self[name] - half, self[name] + half)


def demean(column: np.ndarray, group: np.ndarray, n_groups: int
           ) -> np.ndarray:
    """Subtract each group's own mean: the within transformation."""
    totals = np.bincount(group, weights=column, minlength=n_groups)
    counts = np.bincount(group, minlength=n_groups).astype(float)
    counts[counts == 0] = 1.0
    return column - (totals / counts)[group]


def _demean_matrix(matrix: np.ndarray, group: np.ndarray, n_groups: int
                   ) -> np.ndarray:
    return np.column_stack([demean(matrix[:, j], group, n_groups)
                            for j in range(matrix.shape[1])])


def design(arrays: dict, *, quadratic: bool = False
           ) -> tuple[np.ndarray, tuple[str, ...]]:
    """The regressors: tyre age, race lap, and optionally age squared."""
    columns = [arrays["age"], arrays["lap"]]
    names = ["age", "lap"]
    if quadratic:
        columns.append(arrays["age"] ** 2)
        names.append("age squared")
    return np.column_stack(columns), tuple(names)


def _check_identified(matrix: np.ndarray, names: tuple[str, ...]) -> None:
    """Refuse a design that cannot answer the question.

    If every car in the sample ran a single stint, the age and lap columns
    are the same column after demeaning and the normal equations are
    singular. Solving them anyway would return a number, and the number
    would be meaningless.
    """
    rank = np.linalg.matrix_rank(matrix)
    if rank < matrix.shape[1]:
        raise ValueError(
            "the design is rank deficient after absorbing the fixed "
            f"effects (rank {rank} < {matrix.shape[1]} columns: "
            f"{', '.join(names)}). Tyre age and race lap can only be "
            "separated by cars that made at least one pit stop.")


def _cluster_sandwich(matrix: np.ndarray, residual: np.ndarray,
                      group: np.ndarray, n_groups: int,
                      xtx_inverse: np.ndarray, dof: int) -> np.ndarray:
    """Cluster-robust covariance, clustering on the car-race."""
    k = matrix.shape[1]
    meat = np.zeros((k, k))
    scored = matrix * residual[:, None]
    for g in range(n_groups):
        rows = scored[group == g]
        if rows.size == 0:
            continue
        total = rows.sum(axis=0)
        meat += np.outer(total, total)

    correction = (n_groups / max(n_groups - 1, 1)) * (
        (matrix.shape[0] - 1) / max(dof, 1))
    return correction * xtx_inverse @ meat @ xtx_inverse


def fit_matrix(y: np.ndarray, matrix: np.ndarray, names: tuple[str, ...],
               group: np.ndarray, n_groups: int, *,
               name: str = "within, cluster-robust") -> Fit:
    """Least squares on any design, with the group effects absorbed.

    `fit_within` is this with the standard two columns. It is separate so
    that a model with extra columns — degradation interacted with stint
    position, say — goes through exactly the same estimator and the same
    standard errors rather than a second implementation of them.
    """
    y_within = demean(y, group, n_groups)
    x_within = _demean_matrix(matrix, group, n_groups)
    _check_identified(x_within, names)

    xtx_inverse = np.linalg.inv(x_within.T @ x_within)
    beta = xtx_inverse @ (x_within.T @ y_within)
    residual = y_within - x_within @ beta

    dof = len(y) - n_groups - len(names)
    sigma = float(np.sqrt(residual @ residual / max(dof, 1)))
    covariance = _cluster_sandwich(x_within, residual, group, n_groups,
                                   xtx_inverse, dof)
    total = float(y_within @ y_within)
    explained = 1.0 - float(residual @ residual) / total if total else 0.0

    return Fit(names=names, beta=beta, se=np.sqrt(np.diag(covariance)),
               n=len(y), groups=n_groups, sigma=sigma,
               r_squared=explained, method=name, residuals=residual)


def fit_within(arrays: dict, *, quadratic: bool = False,
               name: str = "within, cluster-robust") -> Fit:
    """Least squares with car-race fixed effects absorbed."""
    matrix, names = design(arrays, quadratic=quadratic)
    return fit_matrix(arrays["seconds"], matrix, names, arrays["group"],
                      arrays["n_groups"], name=name)


def fit_dummy(arrays: dict, *, quadratic: bool = False) -> Fit:
    """The same model with explicit fixed-effect dummies.

    Slower and heavier, and kept only so the within transformation has
    something independent to be checked against.
    """
    y = arrays["seconds"]
    matrix, names = design(arrays, quadratic=quadratic)
    group, n_groups = arrays["group"], arrays["n_groups"]

    dummies = np.zeros((len(y), n_groups))
    dummies[np.arange(len(y)), group] = 1.0
    full = np.column_stack([matrix, dummies])
    beta, *_ = np.linalg.lstsq(full, y, rcond=None)
    residual = y - full @ beta

    dof = len(y) - n_groups - len(names)
    sigma = float(np.sqrt(residual @ residual / max(dof, 1)))
    centred = y - y.mean()
    explained = 1.0 - float(residual @ residual) / float(centred @ centred)
    return Fit(names=names, beta=beta[:len(names)],
               se=np.full(len(names), np.nan), n=len(y), groups=n_groups,
               sigma=sigma, r_squared=explained, method="explicit dummies",
               residuals=residual)


def fit_naive(arrays: dict) -> Fit:
    """Degradation as the within-stint slope on age, with no race-lap term.

    The shortcut this project is about. The fixed effect is per stint
    rather than per car-race, which is the generous version of the
    mistake: it still allows every stint its own level, and it still gets
    the slope wrong, because the omitted race-lap effect is collinear with
    age inside a stint.
    """
    y = arrays["seconds"]
    age = arrays["age"]

    stint_key = arrays["group"].astype(np.int64) * 100 + arrays["stint"]
    _, group = np.unique(stint_key, return_inverse=True)
    n_groups = int(group.max()) + 1

    y_within = demean(y, group, n_groups)
    x_within = demean(age, group, n_groups)[:, None]

    xtx_inverse = np.linalg.inv(x_within.T @ x_within)
    beta = xtx_inverse @ (x_within.T @ y_within)
    residual = y_within - x_within @ beta
    dof = len(y) - n_groups - 1
    sigma = float(np.sqrt(residual @ residual / max(dof, 1)))
    covariance = _cluster_sandwich(x_within, residual, group, n_groups,
                                   xtx_inverse, dof)
    total = float(y_within @ y_within)
    return Fit(names=("age",), beta=beta, se=np.sqrt(np.diag(covariance)),
               n=len(y), groups=n_groups, sigma=sigma,
               r_squared=1.0 - float(residual @ residual) / total,
               method="naive: age only, no race-lap term",
               residuals=residual)


def fit_huber(arrays: dict, *, quadratic: bool = False,
              k: float = HUBER_K) -> Fit:
    """The same within model, robust to laps spent behind another car.

    Traffic adds time to a lap; it never subtracts it, so the contamination
    is one-sided and heavy. Huber weights pull in any lap whose residual is
    more than `k` robust standard deviations out, which keeps the estimate
    from following a handful of blocked laps, without removing slow laps as
    a class — doing that would remove the degradation itself.
    """
    y = arrays["seconds"]
    matrix, names = design(arrays, quadratic=quadratic)
    group, n_groups = arrays["group"], arrays["n_groups"]

    y_within = demean(y, group, n_groups)
    x_within = _demean_matrix(matrix, group, n_groups)
    _check_identified(x_within, names)

    weights = np.ones(len(y))
    beta = np.zeros(len(names))
    for _ in range(HUBER_ITERATIONS):
        weighted = x_within * weights[:, None]
        xtx_inverse = np.linalg.inv(weighted.T @ x_within)
        updated = xtx_inverse @ (weighted.T @ y_within)
        residual = y_within - x_within @ updated
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        scale = max(scale, 1e-9)
        standardised = np.abs(residual) / scale
        weights = np.where(standardised <= k, 1.0, k / standardised)
        if np.max(np.abs(updated - beta)) < HUBER_TOLERANCE:
            beta = updated
            break
        beta = updated

    residual = y_within - x_within @ beta
    weighted = x_within * weights[:, None]
    xtx_inverse = np.linalg.inv(weighted.T @ x_within)
    dof = len(y) - n_groups - len(names)
    covariance = _cluster_sandwich(x_within, residual * weights, group,
                                   n_groups, xtx_inverse, dof)
    total = float(y_within @ y_within)
    return Fit(names=names, beta=beta, se=np.sqrt(np.diag(covariance)),
               n=len(y), groups=n_groups,
               sigma=1.4826 * float(np.median(np.abs(residual))),
               r_squared=1.0 - float(residual @ residual) / total,
               method=f"within, Huber k={k}", weights=weights,
               residuals=residual)


# -- simulation: does any of this recover a truth it was given? -----------

def simulate(n_cars: int = 400, *, degradation: float = 0.060,
             fuel: float = -0.075, race_laps: int = 56,
             stops: int = 2, noise: float = 0.45,
             pace_spread: float = 3.0, seed: int = 0) -> dict:
    """Laps from a world where the two coefficients are known exactly.

    Every car gets its own pace, its own stop laps and its own noise. The
    truth is `degradation` seconds per lap of tyre age and `fuel` seconds
    per lap of race distance; a fitter that cannot return those two numbers
    from this cannot be trusted with the real feed.
    """
    rng = np.random.default_rng(seed)
    seconds, age_col, lap_col, stint_col, group_col = [], [], [], [], []

    for car in range(n_cars):
        pace = 80.0 + pace_spread * rng.standard_normal()
        cuts = np.sort(rng.choice(np.arange(8, race_laps - 7),
                                  size=stops, replace=False))
        starts = np.concatenate(([1], cuts + 1))
        for lap in range(2, race_laps + 1):
            if lap in cuts:                       # in-lap, as in the feed
                continue
            stint = int(np.searchsorted(cuts, lap - 1, side="right"))
            start = starts[stint]
            age = lap - start + 1
            if stint > 0 and age == 1:            # out-lap
                continue
            seconds.append(pace + fuel * lap + degradation * age
                           + noise * rng.standard_normal())
            age_col.append(age)
            lap_col.append(lap)
            stint_col.append(stint + 1)
            group_col.append(car)

    return {"seconds": np.array(seconds), "age": np.array(age_col, float),
            "lap": np.array(lap_col, float),
            "stint": np.array(stint_col, dtype=int),
            "group": np.array(group_col, dtype=int), "n_groups": n_cars,
            "truth": {"degradation": degradation, "fuel": fuel}}


def recovery(levels=(0, 1, 2, 3), **kwargs) -> list[dict]:
    """Recovery at several stop counts, including the unidentified case.

    With no stops the model cannot be fitted at all and must say so. With
    one, age and lap separate but only just. The naive estimate is wrong at
    every level, by about the size of the fuel effect.
    """
    rows = []
    for stops in levels:
        data = simulate(stops=stops, **kwargs)
        truth = data["truth"]
        row = {"stops": stops, **truth}
        try:
            fitted = fit_within(data)
            row["degradation_fit"] = fitted["age"]
            row["fuel_fit"] = fitted["lap"]
            row["degradation_se"] = fitted.error("age")
        except ValueError as error:
            row["degradation_fit"] = float("nan")
            row["fuel_fit"] = float("nan")
            row["degradation_se"] = float("nan")
            row["refused"] = str(error).split("(")[0].strip()
        # The naive fit never refuses, including where nothing is
        # identified — which is exactly the problem with it.
        row["naive"] = fit_naive(data)["age"]
        rows.append(row)
    return rows
