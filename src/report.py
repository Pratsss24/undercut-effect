"""One self-contained HTML file: charts inlined, no assets, no server."""

from __future__ import annotations

import base64
from pathlib import Path

CSS = """
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px;
margin:40px auto;padding:0 20px;color:#16181d}
h1{margin-bottom:4px}h2{margin-top:38px;border-bottom:1px solid #e3e6ea;
padding-bottom:6px}
.sub{color:#5b6472;margin-top:0}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
.tile{flex:1;min-width:150px;background:#f6f8fa;border:1px solid #e3e6ea;
border-radius:9px;padding:14px 16px}
.tile .v{font-size:25px;font-weight:650;letter-spacing:-.5px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:#5b6472;margin-top:3px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f6f8fa;font-weight:600}
td.good{color:#17803d}td.bad{color:#b4232c}
img{width:100%;margin:14px 0;border:1px solid #e3e6ea;border-radius:7px}
.note{background:#fff8e6;border-left:3px solid #d9a441;padding:11px 15px;
margin:18px 0;font-size:14px}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _img(path: Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{Path(path).stem}">'


def build(*, census, attempts, rows, trajectory, split, decay, place,
          horizons, free, implied, lap_fit, naive, full, anchor, late,
          seasons, figures, at_zero, in_excess, out_excess,
          out_path: Path) -> Path:
    tried = [row for row in attempts if not row.control]
    controls = [row for row in attempts if row.control]
    gap_value = implied["seconds per lap of age"]
    gap_low, gap_high = implied.interval("seconds per lap of age")
    lap_low, lap_high = lap_fit.interval("age")
    best = max(rows[1:], key=lambda row: row["against_control"])
    attempts_passed = sum(row["passed"] for row in place
                          if row["advantage"] > 0)
    attempts_total = sum(row["n"] for row in place if row["advantage"] > 0)
    control_share = next(row["share"] for row in place
                         if row["advantage"] == 0)

    tiles = [
        ("undercut attempts", f"{len(tried):,}"),
        ("same-lap controls", f"{len(controls):,}"),
        ("best gain vs control", f"{best['against_control']:+.2f} s"),
        ("attempts that cleared the car",
         f"{attempts_passed / attempts_total:.0%}"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in tiles)

    census_html = "".join(
        f"<tr><td>{key}</td><td>{value:,}</td></tr>"
        for key, value in sorted(census.items(), key=lambda item: -item[1]))

    gain_html = "".join(
        f"<tr><td>{'control (same lap)' if row['advantage'] == 0 else row['advantage']}"
        f"</td><td>{row['n']}</td>"
        f"<td>{row['mean']:+.2f} &plusmn; {1.96 * row['error']:.2f}</td>"
        f'<td class="{"good" if row["against_control"] > 0 else ""}">'
        f"{row['against_control']:+.2f}</td>"
        f"<td>{row['median_age']:.0f}</td></tr>" for row in rows)

    free_html = "".join(
        f"<tr><td>{name}</td><td>{free[name]:+.4f}</td>"
        f"<td>{free.interval(name)[0]:+.4f} to "
        f"{free.interval(name)[1]:+.4f}</td>"
        f"<td>{free.p_value(name):.4f}</td></tr>" for name in free.names)

    decay_html = "".join(
        f"<tr><td>{row['horizon']}</td><td>{row['value']:+.4f}</td>"
        f"<td>{row['low']:+.4f} to {row['high']:+.4f}</td>"
        f"<td>{row['n']:,}</td></tr>" for row in decay)

    place_html = "".join(
        f"<tr><td>{'control (same lap)' if row['advantage'] == 0 else row['advantage']}"
        f"</td><td>{row['n']}</td><td>{row['passed']}</td>"
        f"<td>{row['share']:.1%}</td><td>{row['held']}</td>"
        f"<td>{row['held_share']:.0%}</td></tr>"
        if row["passed"] else
        f"<tr><td>{row['advantage']}</td><td>{row['n']}</td>"
        f"<td>0</td><td>0.0%</td><td>0</td><td>&mdash;</td></tr>"
        for row in place)

    payback_html = "".join(
        f"<tr><td>{name}</td><td>{full[name]:+.3f}</td>"
        f"<td>{full.interval(name)[0]:+.3f} to "
        f"{full.interval(name)[1]:+.3f}</td>"
        f"<td>{full.p_value(name):.4f}</td></tr>" for name in full.names)

    split_html = "".join(
        f"<tr><td>{row['horizon']}</td>"
        f'<td class="good">{row.get("ahead", float("nan")):+.2f}</td>'
        f'<td class="bad">{row.get("behind", float("nan")):+.2f}</td></tr>'
        for row in split if row["horizon"] > 0)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Undercut effect</title><style>{CSS}</style></head><body>
<h1>Undercut effect</h1>
<p class="sub">What pitting first is actually worth, measured from
{"-".join(str(s) for s in seasons)} lap times alone &mdash; with a control
group, because teams do not attempt undercuts at random.</p>

<div class="tiles">{tiles_html}</div>

<div class="note"><b>The gap is in the lap times.</b> Summing a car's lap
times gives its elapsed race time at the end of every lap, so the
difference between two cars' sums <i>is</i> the gap between them. Exact,
no interpolation, no telemetry. Everything here rests on that one
observation.</div>

<h2>Which windows are measured</h2>
{_img(figures['census'])}
<table><tr><th>outcome</th><th>windows</th></tr>{census_html}</table>

<h2>Where the pit-lane time lands</h2>
<p>In this schema a stop recorded on lap <code>L</code> puts
{in_excess:.2f} s of excess on lap <code>L</code> and
{out_excess:.2f} s on lap <code>L+1</code> &mdash; the pit-lane time is
mostly in the out-lap. So on the lap the chased car pits, the chasing car
still reads <b>{at_zero:+.1f} s</b>: a pit loss not yet paid, not an
undercut that failed. Every horizon below is therefore counted from one
lap after the second stop, which is the first point at which both cars
have paid the same toll.</p>

<h2>What it is worth</h2>
{_img(figures['gain'])}
<table><tr><th>laps of head start</th><th>pairs</th>
<th>seconds closed (95%)</th><th>against control</th>
<th>rival tyre age</th></tr>{gain_html}</table>
<p>The control is <b>not zero</b>. A car that is behind is usually behind
because it is slower, and in these windows it loses about two seconds to
the car ahead even when neither gains any tyre advantage. Measured against
zero the undercut looks worthless; measured against the control it is
worth two to four seconds, best at a head start of
{best['advantage']} laps.</p>

<h2>Does the gain run through tyre age?</h2>
<p>The pure-pace model says the whole gain is the rival's tyre age times
the head start, with no free term for the head start on its own: on each
advantage lap the chasing car is quicker by exactly the age penalty, and
that is the only channel. Fitting both terms tests it.</p>
<table><tr><th>term</th><th>estimate</th><th>95% interval</th><th>p</th></tr>
{free_html}</table>
<p>The free head-start term is indistinguishable from zero, so the form
the physics implies is not rejected and the gain can be pushed entirely
through the age channel &mdash; which turns its coefficient into an
estimate of the tyre-age penalty itself.</p>

<h2>A second opinion on tyre degradation</h2>
{_img(figures['estimators'])}
<p>Two estimators of the same quantity, built from different features of
the same season. One regresses a car's lap times on its own tyre age. The
other reads the pace difference between two cars, one on fresh tyres and
one not, over a handful of laps.</p>
<table><tr><th>estimator</th><th>seconds per lap of age</th>
<th>95% interval</th></tr>
<tr><td>gaps between two cars (this project)</td>
<td>{gap_value:+.4f}</td><td>{gap_low:+.4f} to {gap_high:+.4f}</td></tr>
<tr><td>lap times against tyre age (carried over)</td>
<td>{lap_fit['age']:+.4f}</td><td>{lap_low:+.4f} to {lap_high:+.4f}</td></tr>
</table>
<p>The gap-based estimate is
<b>{gap_value / lap_fit['age']:.2f}&times;</b> the lap-time one, which is
the direction a tyre cliff predicts &mdash; a linear fit through stints
that ended before the tyre went away understates the penalty. But the
intervals overlap, so this is <b>agreement between two methods rather
than a contradiction</b>, and one season cannot resolve the difference.</p>

<h2>The pace advantage is a loan</h2>
{_img(figures['decay'])}
<table><tr><th>laps after the rival rejoins</th><th>implied penalty</th>
<th>95% interval</th><th>pairs</th></tr>{decay_html}</table>
{_img(figures['trajectories'])}
<p>Read later and later, the implied penalty decays to nothing. That is
exactly the pure-pace prediction: once the rival has stopped, the chasing
car is the one on the older set, and it hands the gain back at the same
rate it won it. On pace alone, an undercut buys nothing permanent.</p>

<h2>But track position is not handed back</h2>
{_img(figures['position'])}
<table><tr><th>horizon</th><th>cleared the car</th><th>did not</th></tr>
{split_html}</table>
<p>Split by whether the move worked, the two groups behave completely
differently: the cars that got past hold what they won, and the ones that
did not give it all back. The <i>levels</i> in that table are selected -
a pair is in the first group because of its value at the early horizon -
so only the shape is informative, and the shape is tested below with
controls.</p>
<table><tr><th>term</th><th>estimate</th><th>95% interval</th><th>p</th></tr>
{payback_html}</table>
<div class="note"><b>Half the raw effect was composition.</b> Without
controls, getting ahead appears to be worth
{naive['ahead after the move']:+.2f} s of avoided payback. But pairs that
end up ahead started about three seconds apart and pairs that do not
started about twelve, and a pair that starts close is a pair of cars of
similar pace. With the starting gap in the model the figure is
<b>{full['ahead after the move']:+.2f} s</b> &mdash; real, highly
significant, and about half what the raw comparison suggested.</div>

<h2>How often it works at all</h2>
<table><tr><th>laps of head start</th><th>attempts</th><th>cleared</th>
<th>share</th><th>still ahead at {late} laps</th><th>held</th></tr>
{place_html}</table>
<p>Undercutting roughly triples the odds of getting past, from
{control_share:.1%} for a same-lap stop to
{attempts_passed / attempts_total:.1%}. It still fails about
{1 - attempts_passed / attempts_total:.0%} of the time, which is the part
a time-trial model of strategy cannot tell you.</p>

<p class="sub">Generated by <code>run.py</code>. Lap times and pit stops
from the Jolpica mirror of the Ergast schema. The cleaning rules and the
lap-time estimator are carried over from the stint-degradation and
race-strategy-mc projects.</p>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
