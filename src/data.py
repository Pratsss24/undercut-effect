"""Fetch and cache a season of lap times, pit stops and results.

Three endpoints of the open Jolpica mirror of the Ergast schema, none of
which needs a key:

    /{season}/{round}/laps/       every driver's time on every lap
    /{season}/{round}/pitstops/   the lap each stop was made on
    /{season}/{round}/results/    classification, status, grid

The lap endpoint pages at 100 timings, which is the binding constraint:
one race is about 1,100 timings, so eleven requests, and a season is
roughly 270 requests in all. The mirror is rate limited, so the fetcher
pauses between requests and backs off on a 429 rather than hammering it.

**What this feed does not carry: the tyre compound.** Ergast never recorded
it and the mirror inherits that. A stint here is the set of laps between
two pit stops, with no label for what was fitted. Every consequence of that
is in `Scope` in the README; the one that matters most is that degradation
is estimated per circuit and per stint position rather than per compound.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1"
PAGE = 100
PAUSE = 1.1
RETRIES = 5

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"


@dataclass(frozen=True)
class Lap:
    """One driver's time on one lap."""

    season: int
    round: int
    race: str
    circuit: str
    driver: str
    lap: int
    seconds: float
    position: int


@dataclass(frozen=True)
class Stop:
    season: int
    round: int
    driver: str
    lap: int                 # the in-lap: the lap the car came in on
    stop: int
    duration: float


@dataclass(frozen=True)
class Result:
    season: int
    round: int
    race: str
    circuit: str
    driver: str
    team: str
    grid: int
    laps: int
    status: str
    position_text: str

    @property
    def classified(self) -> bool:
        return self.position_text.isdigit()


def parse_lap_time(text: str) -> float | None:
    """'1:39.019' -> 99.019. Returns None for anything unparseable.

    The feed is clean here, but a silent exception in a fetch loop that
    takes five minutes is worth avoiding, and a count of what was dropped
    is worth having.
    """
    text = text.strip()
    if not text:
        return None
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60.0 + float(seconds)
        return float(text)
    except ValueError:
        return None


def _request(url: str) -> dict:
    """GET with backoff on the mirror's rate limit."""
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == RETRIES - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
        except urllib.error.URLError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def _rounds(season: int) -> list[dict]:
    payload = _request(f"{API}/{season}/races/?limit=100")
    return payload["MRData"]["RaceTable"]["Races"]


def fetch_laps(season: int, round_: int, race: str, circuit: str
               ) -> tuple[list[Lap], int]:
    """Every timing for one race, following the 100-row pages."""
    out: list[Lap] = []
    dropped = 0
    offset = 0
    while True:
        payload = _request(f"{API}/{season}/{round_}/laps/"
                           f"?limit={PAGE}&offset={offset}")
        meta = payload["MRData"]
        races = meta["RaceTable"]["Races"]
        if not races:
            break
        for entry in races[0].get("Laps", []):
            lap = int(entry["number"])
            for timing in entry["Timings"]:
                seconds = parse_lap_time(timing["time"])
                if seconds is None:
                    dropped += 1
                    continue
                out.append(Lap(season=season, round=round_, race=race,
                               circuit=circuit, driver=timing["driverId"],
                               lap=lap, seconds=seconds,
                               position=int(timing["position"])))
        offset += PAGE
        if offset >= int(meta["total"]):
            break
        time.sleep(PAUSE)
    return out, dropped


def fetch_stops(season: int, round_: int) -> list[Stop]:
    out: list[Stop] = []
    offset = 0
    while True:
        payload = _request(f"{API}/{season}/{round_}/pitstops/"
                           f"?limit={PAGE}&offset={offset}")
        meta = payload["MRData"]
        races = meta["RaceTable"]["Races"]
        if not races:
            break
        for stop in races[0].get("PitStops", []):
            try:
                duration = float(stop["duration"])
            except (KeyError, ValueError):
                duration = float("nan")
            out.append(Stop(season=season, round=round_,
                            driver=stop["driverId"], lap=int(stop["lap"]),
                            stop=int(stop["stop"]), duration=duration))
        offset += PAGE
        if offset >= int(meta["total"]):
            break
        time.sleep(PAUSE)
    return out


def fetch_results(season: int, round_: int, race: str, circuit: str
                  ) -> list[Result]:
    payload = _request(f"{API}/{season}/{round_}/results/?limit={PAGE}")
    races = payload["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    return [Result(season=season, round=round_, race=race, circuit=circuit,
                   driver=row["Driver"]["driverId"],
                   team=row["Constructor"]["constructorId"],
                   grid=int(row["grid"]), laps=int(row["laps"]),
                   status=row["status"], position_text=row["positionText"])
            for row in races[0]["Results"]]


def fetch_season(season: int, *, verbose: bool = True) -> dict:
    """Everything for one season, in one pass."""
    laps: list[Lap] = []
    stops: list[Stop] = []
    results: list[Result] = []
    dropped = 0

    for race in _rounds(season):
        round_ = int(race["round"])
        name = race["raceName"]
        circuit = race["Circuit"]["circuitId"]
        race_laps, race_dropped = fetch_laps(season, round_, name, circuit)
        if not race_laps:
            if verbose:
                print(f"  {season} r{round_:2d} {name:28s} no lap data")
            continue
        time.sleep(PAUSE)
        race_stops = fetch_stops(season, round_)
        time.sleep(PAUSE)
        race_results = fetch_results(season, round_, name, circuit)
        time.sleep(PAUSE)

        laps.extend(race_laps)
        stops.extend(race_stops)
        results.extend(race_results)
        dropped += race_dropped
        if verbose:
            print(f"  {season} r{round_:2d} {name:28s} "
                  f"{len(race_laps):5d} laps, {len(race_stops):3d} stops")

    return {"laps": [asdict(x) for x in laps],
            "stops": [asdict(x) for x in stops],
            "results": [asdict(x) for x in results],
            "dropped": dropped}


def load(seasons: tuple[int, ...] = (2023,), *, refresh: bool = False
         ) -> tuple[list[Lap], list[Stop], list[Result]]:
    """Cached per season, so a re-run costs nothing."""
    CACHE.mkdir(exist_ok=True)
    laps: list[Lap] = []
    stops: list[Stop] = []
    results: list[Result] = []

    for season in seasons:
        path = CACHE / f"season_{season}.json"
        if refresh or not path.exists():
            print(f"fetching {season} (a few minutes)")
            payload = fetch_season(season)
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        laps.extend(Lap(**row) for row in payload["laps"])
        stops.extend(Stop(**row) for row in payload["stops"])
        results.extend(Result(**row) for row in payload["results"])

    return laps, stops, results
