"""What the stations of a region show together: bursts in the same minute, and loudness against the neighbours.

    uv run python -m forecasting_earthquakes_hopefully.together MINUTES_DIR LOUDNESS_CSV OUT.csv

Fixed on 2026-11-22 before any score was read. A day of a region counts when at least MIN_STATIONS of its
stations have a usable day (bursts.py) or a usable window (LOUDNESS_CSV of windows.py).

  coin_1, _7, _28, _rel   minutes of the day in which at least MIN_STATIONS stations have a burst (bursts.py) in
                          that same minute, per 1440 minutes valid at that many stations, turned into the four
                          columns of bursts.features. One station's burst is traffic; several at once is an
                          earthquake or something else that reaches them all. The same for every station of the
                          region. No allowance for travel time: an event split over two minutes can be missed.
  rel_<band>              the station's loud_<band> minus the median of it over the other stations of the region
                          on that date: what changed here and not around
  common_<band>           that median: what changed everywhere (weather, the sea), kept as a control

  OUT.csv              station, date and the columns, for the days where all of them exist
  OUT.provenance.json  command line, parameters, commit, hashes of the inputs

Exits nonzero when no row is written.
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import bursts, split
from .minutes import MINUTES
from .survey import commit

MIN_STATIONS = 3


def coincident(masks):
    """masks: [(burst (1440,), valid (1440,))] of the stations of one day -> coincident minutes per 1440 minutes
    valid at MIN_STATIONS stations or more; NaN when there are fewer stations or no such minute."""
    if len(masks) < MIN_STATIONS:
        return np.nan
    burst, valid = np.stack([m[0] for m in masks]), np.stack([m[1] for m in masks])
    enough = valid.sum(axis=0) >= MIN_STATIONS
    return (burst.sum(axis=0) >= MIN_STATIONS)[enough].sum() / enough.sum() * MINUTES if enough.any() else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("minutes", type=Path)
    ap.add_argument("loudness", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    per_day = {}                                             # (region, date) -> [(burst, valid)]
    for f in sorted(args.minutes.glob("*.npz")):
        z = np.load(f)
        burst, valid, ok = bursts.burst_minutes(z["x"])
        for i in np.where(ok)[0]:
            per_day.setdefault((split.region_of(f.stem), dt.date.fromisoformat(str(z["dates"][i]))), []).append((burst[i], valid[i]))
    coin = {}
    for region in sorted({k[0] for k in per_day}):
        days = sorted(d for r, d in per_day if r == region)
        rate = [coincident(per_day[region, d]) for d in days]
        coin[region] = bursts.features(days, rate)
        print(f"{region}: {len(coin[region])} days with {MIN_STATIONS} stations or more, median "
              f"{np.nanmedian(rate):.1f} coincident minutes a day", flush=True)
    loud = pd.read_csv(args.loudness)
    bands = [c for c in loud.columns if c.startswith("loud_")]
    loud["region"] = [split.region_of(s) for s in loud["station"]]
    rows = []
    for (region, date), part in loud.groupby(["region", "date"]):
        c = coin.get(region, {}).get(dt.date.fromisoformat(date))
        if c is None or len(part) < MIN_STATIONS:
            continue
        v = part[bands].to_numpy()
        for i, station in enumerate(part["station"]):
            others = np.median(np.delete(v, i, axis=0), axis=0)
            rows.append((station, date, *c, *(v[i] - others), *others))
    if not rows:
        sys.exit("no day with enough stations: nothing written")
    cols = ([f"coin_{k}" for k in ("1", "7", "28", "rel")] + [b.replace("loud_", "rel_") for b in bands]
            + [b.replace("loud_", "common_") for b in bands])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["station", "date"] + cols).to_csv(args.out, index=False, lineterminator="\n")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prov = args.minutes / "provenance.json"
    args.out.with_suffix(".provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(min_stations=MIN_STATIONS, factor=bursts.FACTOR, min_day=bursts.MIN_DAY),
        inputs={str(p): sha(p) for p in (args.loudness, prov) if Path(p).exists()},
        versions=dict(numpy=np.__version__, pandas=pd.__version__), rows=len(rows)), indent=1))
    print(f"{len(rows)} station-days -> {args.out}")


if __name__ == "__main__":
    main()
