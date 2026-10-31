"""What has to hold for any of these numbers to mean anything.

Three kinds of anchor, in descending order of worth.

**Arithmetic.** The gap between two cars is a difference of cumulative lap
times, so it can be checked against hand-built cases where the answer is
known by construction. No tolerance, no judgement.

**Recovery of a planted effect.** A synthetic race is built with a known
tyre-age penalty and a known undercut, run through the real pipeline, and
the estimator is required to find what was put in. If it cannot do that, it
cannot be trusted on the season.

**Refusal.** The estimator must decline a rank-deficient design and must
not silently treat a missing lap as a zero gap, because both would return
a confident number from nothing.

The real-data tests pin the headline figures. They skip rather than fetch:
run `python run.py` once to cache the season.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data           # noqa: E402
import effect         # noqa: E402
import gaps           # noqa: E402
import model          # noqa: E402
import pairs          # noqa: E402
import stints         # noqa: E402
from data import Lap, Result, Stop   # noqa: E402


@pytest.fixture(scope="module")
def feed():
    """The cached season. The suite never goes to the network itself."""
    if not (ROOT / "data" / "season_2023.json").exists():
        pytest.skip("run `python run.py` once to cache the season")
    return data.load()


@pytest.fixture(scope="module")
def attempts(feed):
    built, _ = pairs.build(*feed)
    return built


# -------------------------------------------------------------- the gap

def _lap(round_, driver, lap, seconds):
    return Lap(season=2023, round=round_, race="Test", circuit="test",
               driver=driver, lap=lap, seconds=seconds, position=1)


def test_the_gap_is_a_difference_of_cumulative_times():
    """A car lapping 90.0 against one lapping 91.0 falls one second
    behind per lap, exactly."""
    rows = ([_lap(1, "fast", n, 90.0) for n in range(1, 11)]
            + [_lap(1, "slow", n, 91.0) for n in range(1, 11)])
    times = gaps.elapsed(rows)
    fast, slow = (2023, 1, "fast"), (2023, 1, "slow")
    for lap in range(1, 11):
        assert gaps.gap(times, fast, slow, lap) == pytest.approx(float(lap))
    assert gaps.gap(times, slow, fast, 10) == pytest.approx(-10.0)


def test_the_gap_is_zero_at_lap_zero():
    rows = [_lap(1, "a", 1, 90.0), _lap(1, "b", 1, 95.0)]
    times = gaps.elapsed(rows)
    assert gaps.gap(times, (2023, 1, "a"), (2023, 1, "b"), 0) == 0.0


def test_a_missing_lap_poisons_the_car_rather_than_one_lap():
    """A hole in the middle would silently shorten every later elapsed
    time, which would look like a car that gained twenty seconds. Better
    to lose the car than to report that."""
    rows = [_lap(1, "a", n, 90.0) for n in (1, 2, 4, 5)]
    times = gaps.elapsed(rows)
    assert np.isnan(times[(2023, 1, "a")][5])


def test_an_unrun_lap_is_not_a_zero_gap():
    rows = [_lap(1, "a", n, 90.0) for n in range(1, 11)] + \
           [_lap(1, "b", n, 90.0) for n in range(1, 6)]
    times = gaps.elapsed(rows)
    assert np.isnan(gaps.gap(times, (2023, 1, "a"), (2023, 1, "b"), 9))


def test_prior_pace_reads_the_drift_it_is_given():
    rows = ([_lap(1, "fast", n, 90.0) for n in range(1, 21)]
            + [_lap(1, "slow", n, 90.25) for n in range(1, 21)])
    times = gaps.elapsed(rows)
    measured = gaps.pace_difference(times, (2023, 1, "fast"),
                                    (2023, 1, "slow"), 5, 15)
    assert measured == pytest.approx(0.25)


# ------------------------------------------------------- least squares

def test_the_estimator_recovers_a_planted_line():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 2.5 + 1.75 * x + rng.normal(0, 0.1, 400)
    fit = effect.ols(y, np.column_stack([np.ones(400), x]),
                     ("const", "slope"), np.arange(400))
    assert fit["const"] == pytest.approx(2.5, abs=0.03)
    assert fit["slope"] == pytest.approx(1.75, abs=0.03)


def test_clustering_widens_the_interval_when_rows_are_dependent():
    """Rows that share a shock carry less information than their count
    suggests, and the error has to say so."""
    rng = np.random.default_rng(1)
    group = np.repeat(np.arange(40), 20)
    shock = rng.normal(0, 2.0, 40)[group]
    x = rng.normal(size=800)
    y = 1.0 + 0.5 * x + shock + rng.normal(0, 0.3, 800)
    design = np.column_stack([np.ones(800), x])
    clustered = effect.ols(y, design, ("const", "slope"), group)
    independent = effect.ols(y, design, ("const", "slope"), np.arange(800))
    assert clustered.error("const") > 2.0 * independent.error("const")


def test_a_rank_deficient_design_is_refused():
    x = np.arange(50.0)
    with pytest.raises(ValueError, match="rank deficient"):
        effect.ols(x, np.column_stack([np.ones(50), x, 2 * x]),
                   ("const", "x", "twice x"), np.arange(50))


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="different lengths"):
        effect.ols(np.zeros(10), np.ones((9, 1)), ("const",), np.arange(10))


# ----------------------------------------------- recovery on built data

def _synthetic(*, degradation: float, advantage: int, laps: int = 40,
               stop: int = 20, rival_age_at_stop: int = 20,
               base: float = 90.0, loss: float = 21.0, cars: int = 60,
               noise: float = 0.0, offset: float = 3.0, fillers: int = 10,
               seed: int = 0):
    """A race where the undercut is worth exactly what the physics says.

    Pairs of cars run identical pace except for a tyre-age penalty. One
    pits on `stop`, the other `advantage` laps later. The chasing car
    starts `offset` seconds back and is otherwise identical, because a
    pair that is exactly level is not a pair anybody is undercutting and
    the filters say so.

    Nothing else differs, so the measured gain has one correct value -
    `degradation * rival_age_at_stop * advantage` - and the pipeline
    either finds it or does not.

    `fillers` are cars that run a constant lap and never pit. They exist
    because the safety car detector reads the *field* median, and in a
    two-car field the lap containing a twenty-second pit loss is the
    median, so the whole window gets flagged as neutralised and thrown
    away. That is the detector behaving correctly on real races and
    badly on a field of two; the fixture owes it a field.
    """
    rng = np.random.default_rng(seed)
    rows: list[Lap] = []
    stops: list[Stop] = []
    results: list[Result] = []

    for pair in range(cars // 2):
        round_ = pair + 1
        plan = {"ahead": stop + advantage, "behind": stop}
        for name, own_stop in plan.items():
            elapsed_age = rival_age_at_stop - stop
            for lap in range(1, laps + 1):
                age = (lap + elapsed_age if lap <= own_stop
                       else lap - own_stop)
                seconds = base + degradation * age
                if lap == 1 and name == "behind":
                    seconds += offset
                if lap == own_stop + 1:
                    seconds += loss
                if noise:
                    seconds += rng.normal(0, noise)
                rows.append(Lap(season=2023, round=round_, race="Synthetic",
                                circuit="synthetic", driver=name, lap=lap,
                                seconds=seconds, position=1))
            stops.append(Stop(season=2023, round=round_, driver=name,
                              lap=own_stop, stop=1, duration=float("nan")))
            results.append(Result(
                season=2023, round=round_, race="Synthetic",
                circuit="synthetic", driver=name, team="t", grid=1,
                laps=laps, status="Finished", position_text="1"))

        for filler in range(fillers):
            name = f"filler{filler}"
            for lap in range(1, laps + 1):
                rows.append(Lap(season=2023, round=round_, race="Synthetic",
                                circuit="synthetic", driver=name, lap=lap,
                                seconds=base + 40.0 * (lap == 1),
                                position=1))
            results.append(Result(
                season=2023, round=round_, race="Synthetic",
                circuit="synthetic", driver=name, team="t", grid=1,
                laps=laps, status="Finished", position_text="1"))
    return rows, stops, results


def test_the_pipeline_recovers_a_planted_tyre_penalty():
    """The central claim, where the truth is known. Built with a 0.05
    s/lap penalty and a three-lap head start, the gap-based estimator has
    to return 0.05."""
    built = []
    for advantage in (1, 2, 3, 4, 0):
        rows, stops, results = _synthetic(
            degradation=0.05, advantage=advantage, seed=advantage)
        found, _ = pairs.build(rows, stops, results)
        built += found
    assert built, "the synthetic season produced no measurable pairs"
    columns = pairs.arrays(built, pairs.HORIZONS.index(1))
    fit = effect.degradation_from_gaps(columns, 1, drift=False)
    assert fit["seconds per lap of age"] == pytest.approx(0.05, abs=0.004)


def test_a_planted_race_with_no_degradation_shows_no_undercut():
    """If tyres do not go off, pitting early buys nothing, and the
    estimator must say so rather than finding an effect in the pit loss."""
    built = []
    for advantage in (0, 2, 4):
        rows, stops, results = _synthetic(
            degradation=0.0, advantage=advantage, seed=advantage + 10)
        found, _ = pairs.build(rows, stops, results)
        built += found
    columns = pairs.arrays(built, pairs.HORIZONS.index(1))
    fit = effect.degradation_from_gaps(columns, 1, drift=False)
    assert abs(fit["seconds per lap of age"]) < 0.004


def test_the_pit_loss_cancels_once_both_cars_have_stopped():
    """Doubling the pit loss changes nothing about the measured gain,
    because both cars pay it. If this failed, the whole measurement would
    be reading a pit loss."""
    found = {}
    for loss in (15.0, 30.0):
        built = []
        for advantage in (0, 2, 4):
            rows, stops, results = _synthetic(
                degradation=0.05, advantage=advantage, loss=loss,
                seed=advantage)
            part, _ = pairs.build(rows, stops, results)
            built += part
        columns = pairs.arrays(built, pairs.HORIZONS.index(1))
        found[loss] = effect.degradation_from_gaps(
            columns, 1, drift=False)["seconds per lap of age"]
    assert found[15.0] == pytest.approx(found[30.0], abs=1e-6)


def test_the_gain_is_still_unpaid_on_the_lap_the_rival_pits():
    """Horizon zero sits between the two pit losses, so it must look
    terrible. This is the convention that fixes the anchor at h=1."""
    rows, stops, results = _synthetic(degradation=0.05, advantage=3)
    built, _ = pairs.build(rows, stops, results)
    zero = pairs.HORIZONS.index(0)
    values = [row.deltas[zero] for row in built if row.advantage == 3]
    assert values and np.mean(values) < -10.0


# --------------------------------------------------------------- filters

def test_a_pair_too_far_apart_is_not_an_undercut_attempt():
    rows, stops, results = _synthetic(degradation=0.05, advantage=2)
    tight, _ = pairs.build(rows, stops, results, max_gap=0.5)
    assert not tight


def test_a_head_start_beyond_the_limit_is_dropped():
    rows, stops, results = _synthetic(degradation=0.05, advantage=5)
    built, census = pairs.build(rows, stops, results, max_advantage=3)
    assert not [row for row in built if row.advantage == 5]
    assert census.get("dropped: head start too long to be an undercut", 0) > 0


def test_same_lap_pairs_are_marked_as_controls():
    rows, stops, results = _synthetic(degradation=0.05, advantage=0)
    built, _ = pairs.build(rows, stops, results)
    assert built and all(row.control for row in built)
    assert all(row.advantage == 0 for row in built)


# ------------------------------------------------------------- real data

def test_the_season_yields_both_attempts_and_controls(attempts):
    tried = [row for row in attempts if not row.control]
    controls = [row for row in attempts if row.control]
    assert len(tried) > 500
    assert len(controls) > 100


def test_the_control_group_is_not_zero(attempts):
    """The reason the control exists. A car that is behind is usually
    behind because it is slower, so the naive baseline of zero would
    flatter every undercut by about two seconds."""
    rows = effect.by_advantage(attempts, pairs.HORIZONS, 1)
    control = rows[0]
    assert control["advantage"] == 0
    assert control["mean"] < -1.0


def test_the_undercut_beats_the_control_at_every_head_start(attempts):
    rows = effect.by_advantage(attempts, pairs.HORIZONS, 1)
    gains = [row["against_control"] for row in rows if row["advantage"] > 0]
    assert all(value > 1.5 for value in gains)
    assert max(gains) < 6.0


def test_all_the_gain_runs_through_tyre_age(attempts):
    """The physics says there is no channel for a head start except the
    rival's tyre age. Adding a free term for it finds nothing, which is
    the functional form surviving a test it could have failed."""
    columns = pairs.arrays(attempts, pairs.HORIZONS.index(1))
    fit = effect.gain(columns, 1)
    low, high = fit.interval("per lap of head start")
    assert low < 0 < high
    assert fit.p_value("per lap of head start x age") < 0.15


def test_the_two_estimators_of_degradation_agree(feed, attempts):
    """Different features of the same season, and the intervals overlap.
    The gap-based figure is higher, which is the direction a tyre cliff
    predicts, but one season cannot separate them and this asserts the
    overlap rather than the difference."""
    columns = pairs.arrays(attempts, pairs.HORIZONS.index(1))
    from_gaps = effect.degradation_from_gaps(columns, 1)
    gap_low, gap_high = from_gaps.interval("seconds per lap of age")

    clean, _ = stints.build(*feed)
    lap_fit = model.fit_within(stints.arrays(clean))
    lap_low, lap_high = lap_fit.interval("age")

    assert gap_low < lap_high and lap_low < gap_high      # they overlap
    assert from_gaps["seconds per lap of age"] > lap_fit["age"]
    assert from_gaps["seconds per lap of age"] < 2.0 * lap_fit["age"]


def test_the_pace_advantage_decays_to_nothing(attempts):
    """Read far enough after the rival rejoins, the implied penalty is
    no longer distinguishable from zero: the gain has been repaid."""
    early = effect.degradation_from_gaps(
        pairs.arrays(attempts, pairs.HORIZONS.index(1)), 1)
    late = effect.degradation_from_gaps(
        pairs.arrays(attempts, pairs.HORIZONS.index(12)), 12)
    assert late["seconds per lap of age"] < \
        0.6 * early["seconds per lap of age"]
    low, high = late.interval("seconds per lap of age")
    assert low < 0 < high


def test_getting_ahead_avoids_the_payback(attempts):
    """The track-position effect, which the single-car strategy model
    could not represent at all."""
    fit = effect.payback(attempts, pairs.HORIZONS, early=1, late=12)
    low, high = fit.interval("ahead after the move")
    assert low > 1.5
    assert fit.p_value("ahead after the move") < 0.01


def test_controlling_for_the_starting_gap_roughly_halves_that_effect(
        attempts):
    """Kept as a test because the first version of this analysis reported
    the uncontrolled number. Pairs that end up ahead started far closer
    together, and that is composition rather than mechanism."""
    naive = effect.payback(attempts, pairs.HORIZONS, early=1, late=12,
                           controls=False)
    full = effect.payback(attempts, pairs.HORIZONS, early=1, late=12)
    assert naive["ahead after the move"] > 1.5 * full["ahead after the move"]
    assert full["ahead after the move"] > 2.0


def test_undercutting_beats_a_same_lap_stop_for_track_position(attempts):
    place = effect.positions(attempts, pairs.HORIZONS, 1)
    control = next(row for row in place if row["advantage"] == 0)
    tried = [row for row in place if row["advantage"] > 0]
    passed = sum(row["passed"] for row in tried)
    total = sum(row["n"] for row in tried)
    assert passed / total > 2.0 * control["share"]
    # And it is still a long shot.
    assert passed / total < 0.2


def test_lap_time_parsing():
    assert data.parse_lap_time("1:33.996") == pytest.approx(93.996)
    assert data.parse_lap_time("") is None


def test_a_constant_outcome_has_no_r_squared_rather_than_zero():
    """There is no variance to explain, so the answer is undefined. A
    zero here would read as a model that explains nothing, which is a
    different and wrong claim."""
    fit = effect.ols(np.ones(40), np.column_stack([np.ones(40),
                                                   np.arange(40.0)]),
                     ("const", "x"), np.arange(40))
    assert np.isnan(fit.r_squared)


def test_the_gap_is_antisymmetric():
    rows = ([_lap(1, "a", n, 90.0) for n in range(1, 6)]
            + [_lap(1, "b", n, 92.0) for n in range(1, 6)])
    times = gaps.elapsed(rows)
    a, b = (2023, 1, "a"), (2023, 1, "b")
    for lap in range(1, 6):
        assert gaps.gap(times, a, b, lap) == -gaps.gap(times, b, a, lap)


def test_a_neutralised_window_is_thrown_away(feed):
    """Pit losses fall by two thirds behind a safety car, so a window
    containing one is measuring a different quantity entirely."""
    _, census = pairs.build(*feed)
    assert census["dropped: neutralised inside the window"] > 500


def test_a_lapped_car_cannot_enter_a_pair(attempts):
    """Nothing detects a lap down directly; the maximum-gap filter does
    it, because a car a lap behind shows a gap of a whole lap time."""
    assert all(0.0 < row.gap_before <= pairs.MAX_GAP for row in attempts)
    assert max(row.gap_before for row in attempts) < 60.0


def test_every_attempt_has_the_head_start_it_claims(attempts):
    assert all(0 <= row.advantage <= pairs.MAX_ADVANTAGE
               for row in attempts)
    assert all(row.control == (row.advantage == 0) for row in attempts)


def test_the_car_undercutting_is_the_one_on_older_tyres(attempts):
    """Written expecting the opposite, which was wrong and is worth
    keeping. A team does not pit because the car ahead is struggling; it
    pits because its own tyres are gone. Across 2023 the undercutting
    car averages about a lap more tyre age than the car it is chasing,
    and the margin grows with the head start taken.

    That matters for reading the rest: the move is often forced rather
    than chosen, so it is less cherry-picked than a story about teams
    waiting for the perfect moment would suggest.
    """
    tried = [row for row in attempts if not row.control]
    own = np.mean([row.own_age for row in tried])
    rival = np.mean([row.rival_age for row in tried])
    assert own > rival
    assert 0.5 < own - rival < 3.0
