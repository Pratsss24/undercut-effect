"""Build the report as a printable PDF.

`paper.py` owns the page furniture; everything here is this project's own
argument, in the order the argument runs.
"""

from __future__ import annotations

from pathlib import Path

from paper import Paper

SOURCE = ("Lap times and pit stops from the Jolpica mirror of the Ergast "
          "schema. The cleaning rules and the lap-time estimator are "
          "carried over from the stint-degradation and race-strategy-mc "
          "projects.")


def _advantage(value) -> str:
    return "control (same lap)" if value == 0 else str(value)


def _census(census: dict) -> tuple[list[str], list[list[str]]]:
    rows = [[key, f"{value:,}"] for key, value
            in sorted(census.items(), key=lambda item: -item[1])]
    return (["outcome", "windows"], rows)


def _gain(rows) -> tuple[list[str], list[list[str]]]:
    out = [[_advantage(one["advantage"]), str(one["n"]),
            f"{one['mean']:+.2f} ± {1.96 * one['error']:.2f}",
            f"{one['against_control']:+.2f}", f"{one['median_age']:.0f}"]
           for one in rows]
    return (["laps of head start", "pairs", "seconds closed (95%)",
             "against control", "rival tyre age"], out)


def _terms(fit, digits: int = 4) -> tuple[list[str], list[list[str]]]:
    rows = []
    for name in fit.names:
        low, high = fit.interval(name)
        rows.append([name, f"{fit[name]:+.{digits}f}",
                     f"{low:+.{digits}f} to {high:+.{digits}f}",
                     f"{fit.p_value(name):.4f}"])
    return (["term", "estimate", "95% interval", "p"], rows)


def _decay(decay) -> tuple[list[str], list[list[str]]]:
    rows = [[str(one["horizon"]), f"{one['value']:+.4f}",
             f"{one['low']:+.4f} to {one['high']:+.4f}", f"{one['n']:,}"]
            for one in decay]
    return (["laps after the rival rejoins", "implied penalty",
             "95% interval", "pairs"], rows)


def _split(split) -> tuple[list[str], list[list[str]]]:
    nan = float("nan")
    rows = [[str(one["horizon"]), f"{one.get('ahead', nan):+.2f}",
             f"{one.get('behind', nan):+.2f}"]
            for one in split if one["horizon"] > 0]
    return (["horizon", "cleared the car", "did not"], rows)


def _place(place, late) -> tuple[list[str], list[list[str]]]:
    rows = []
    for one in place:
        if one["passed"]:
            rows.append([_advantage(one["advantage"]), str(one["n"]),
                         str(one["passed"]), f"{one['share']:.1%}",
                         str(one["held"]), f"{one['held_share']:.0%}"])
        else:
            rows.append([str(one["advantage"]), str(one["n"]), "0", "0.0%",
                         "0", "—"])
    return (["laps of head start", "attempts", "cleared", "share",
             f"still ahead at {late} laps", "held"], rows)


def build(*, census, attempts, rows, trajectory, split, decay, place,
          horizons, free, implied, lap_fit, naive, full, anchor, late,
          seasons, figures, at_zero, in_excess, out_excess,
          out_path: Path) -> Path:
    tried = [one for one in attempts if not one.control]
    controls = [one for one in attempts if one.control]
    gap_value = implied["seconds per lap of age"]
    gap_low, gap_high = implied.interval("seconds per lap of age")
    lap_low, lap_high = lap_fit.interval("age")
    best = max(rows[1:], key=lambda one: one["against_control"])
    passed = sum(one["passed"] for one in place if one["advantage"] > 0)
    total = sum(one["n"] for one in place if one["advantage"] > 0)
    control_share = next(one["share"] for one in place
                         if one["advantage"] == 0)
    out = Path(out_path)
    out = out.with_name("report.pdf") if out.suffix != ".pdf" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    with Paper(out, "Undercut effect",
               f"What pitting first is actually worth, measured from "
               f"{'-'.join(str(one) for one in seasons)} lap times alone "
               f"— with a control group, because teams do not attempt "
               f"undercuts at random.") as paper:

        paper.cover(
            headline=f"Worth {best['against_control']:+.2f} s against a "
                     f"control, not against zero",
            small=f"Best at a head start of {best['advantage']} laps, over "
                  f"{len(tried):,} attempts against {len(controls):,} "
                  f"same-lap controls. It clears the car "
                  f"{passed / total:.0%} of the time.",
            tiles=[("undercut attempts", f"{len(tried):,}"),
                   ("same-lap controls", f"{len(controls):,}"),
                   ("best gain vs control",
                    f"{best['against_control']:+.2f} s"),
                   ("attempts that cleared the car",
                    f"{passed / total:.0%}")],
            blurb="The gap is in the lap times. Summing a car's lap times "
                  "gives its elapsed race time at the end of every lap, so "
                  "the difference between two cars' sums is the gap between "
                  "them. Exact, no interpolation, no telemetry. Everything "
                  "here rests on that one observation.",
            figure_path=figures["census"])

        paper.section(
            "Which windows are measured", table=_census(census),
            after=f"In this schema a stop recorded on lap L puts "
                  f"{in_excess:.2f} s of excess on lap L and "
                  f"{out_excess:.2f} s on lap L+1 — the pit-lane time "
                  f"is mostly in the out-lap. So on the lap the chased car "
                  f"pits, the chasing car still reads {at_zero:+.1f} s: a "
                  f"pit loss not yet paid, not an undercut that failed. "
                  f"Every horizon below is counted from one lap after the "
                  f"second stop, the first point at which both cars have "
                  f"paid the same toll.")

        paper.section(
            "What it is worth",
            warn=f"The control is not zero. A car that is behind is usually "
                 f"behind because it is slower, and in these windows it "
                 f"loses about two seconds to the car ahead even when "
                 f"neither gains any tyre advantage. Measured against zero "
                 f"the undercut looks worthless; measured against the "
                 f"control it is worth two to four seconds, best at a head "
                 f"start of {best['advantage']} laps.",
            figure_path=figures["gain"], table=_gain(rows))

        paper.section(
            "Does the gain run through tyre age?",
            note="The pure-pace model says the whole gain is the rival's "
                 "tyre age times the head start, with no free term for the "
                 "head start on its own. Fitting both terms tests it.",
            body="The free head-start term is indistinguishable from zero, "
                 "so the form the physics implies is not rejected and the "
                 "gain can be pushed entirely through the age channel "
                 "— which turns its coefficient into an estimate of "
                 "the tyre-age penalty itself.",
            table=_terms(free))

        paper.section(
            "A second opinion on tyre degradation",
            note="Two estimators of the same quantity, built from different "
                 "features of the same season: one regresses a car's lap "
                 "times on its own tyre age, the other reads the pace "
                 "difference between two cars over a handful of laps.",
            warn=f"The gap-based estimate is "
                 f"{gap_value / lap_fit['age']:.2f}× the lap-time one, "
                 f"which is the direction a tyre cliff predicts — a "
                 f"linear fit through stints that ended before the tyre "
                 f"went away understates the penalty. But the intervals "
                 f"overlap, so this is agreement between two methods rather "
                 f"than a contradiction, and one season cannot resolve the "
                 f"difference.",
            figure_path=figures["estimators"],
            table=(["estimator", "seconds per lap of age", "95% interval"],
                   [["gaps between two cars (this project)",
                     f"{gap_value:+.4f}",
                     f"{gap_low:+.4f} to {gap_high:+.4f}"],
                    ["lap times against tyre age (carried over)",
                     f"{lap_fit['age']:+.4f}",
                     f"{lap_low:+.4f} to {lap_high:+.4f}"]]))

        paper.section(
            "The pace advantage is a loan",
            body="Read later and later, the implied penalty decays to "
                 "nothing. That is exactly the pure-pace prediction: once "
                 "the rival has stopped, the chasing car is the one on the "
                 "older set, and it hands the gain back at the same rate it "
                 "won it. On pace alone, an undercut buys nothing "
                 "permanent.",
            figure_path=[figures["decay"], figures["trajectories"]],
            table=_decay(decay))

        paper.section(
            "But track position is not handed back",
            body="Split by whether the move worked, the two groups behave "
                 "completely differently: the cars that got past hold what "
                 "they won, and the ones that did not give it all back. The "
                 "levels in this table are selected — a pair is in the "
                 "first group because of its value at the early horizon "
                 "— so only the shape is informative, and the shape is "
                 "tested with controls below.",
            figure_path=figures["position"], table=_split(split))

        paper.section(
            "What the controls do to it",
            warn=f"Half the raw effect was composition. Without controls, "
                 f"getting ahead appears to be worth "
                 f"{naive['ahead after the move']:+.2f} s of avoided "
                 f"payback. But pairs that end up ahead started about three "
                 f"seconds apart and pairs that do not started about "
                 f"twelve, and a pair that starts close is a pair of cars "
                 f"of similar pace. With the starting gap in the model the "
                 f"figure is {full['ahead after the move']:+.2f} s — "
                 f"real, highly significant, and about half what the raw "
                 f"comparison suggested.",
            table=_terms(full, digits=3))

        paper.section(
            "How often it works at all",
            body=f"Undercutting roughly triples the odds of getting past, "
                 f"from {control_share:.1%} for a same-lap stop to "
                 f"{passed / total:.1%}. It still fails about "
                 f"{1 - passed / total:.0%} of the time, which is the part "
                 f"a time-trial model of strategy cannot tell you.",
            table=_place(place, late), after=SOURCE)

    return out
