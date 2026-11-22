"""Short bursts counted in the station's own recording: small earthquakes below what a catalogue lists.

    uv run python -m forecasting_earthquakes_hopefully.bursts MINUTES_DIR OUT.csv

A minute is a burst when, in the 2-5 Hz or the 5-10 Hz band, the maximum of the envelope is at least FACTOR
times the background on all three components at once. The background of a component and band is the median
over the day of the per-minute mean. The rate of a day is bursts per 1440 valid minutes; a day needs MIN_DAY of
its minutes valid on every component. The definition was fixed on 2026-11-22 before any score was read; it
counts anything short and broadband on three components, traffic and blasts included, which is why the same
count over the local night (NIGHT, UTC hours) is written beside it.

For a station and a day d, from days up to d only (days without data are left out of the means):
  <name>_1, _7, _28   log(1 + mean rate) of day d, of d-6..d and of d-27..d
  <name>_rel          log(1 + mean rate of d-13..d) - log(1 + mean rate of d-27..d-14); 0 without those days
with <name> bursts (all hours) and night. rivals.py --features turns each name into one forecast.

  OUT.csv              station, date and the eight columns, one row per day that has a rate
  OUT.provenance.json  command line, parameters, commit, hash of the summaries' provenance

Exits nonzero when no row is written.
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .envelopes import BANDS
from .minutes import COMPONENTS, MINUTES
from .survey import commit

FACTOR = 10.0
BURST_BANDS = (2, 3)                 # 2-5 Hz and 5-10 Hz in envelopes.BANDS
MIN_DAY = 0.5
NIGHT = (19, 1)                      # 19:00 to 01:00 UTC, 22:00 to 04:00 in Türkiye
NB, NC = len(BANDS), len(COMPONENTS)


def burst_minutes(x):
    """x (n, 2 * 3 * bands, 1440) -> (burst, valid, day_ok): the burst minutes (n, 1440), the minutes with data
    on every component, and the days (n,) with enough of them."""
    mean = x[:, :NC * NB].reshape(len(x), NC, NB, MINUTES)[:, :, BURST_BANDS]
    peak = x[:, NC * NB:].reshape(len(x), NC, NB, MINUTES)[:, :, BURST_BANDS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        background = np.nanmedian(mean, axis=3, keepdims=True)
    valid = np.isfinite(x[:, :NC * NB]).all(axis=1)
    burst = ((peak - background >= np.log10(FACTOR)).all(axis=1).any(axis=1)) & valid
    return burst, valid, valid.mean(axis=1) >= MIN_DAY


def day_rates(x):
    """x (n, 2 * 3 * bands, 1440) -> bursts per 1440 valid minutes, all hours and night: two (n,) arrays, NaN
    for a day with too little data."""
    burst, valid, ok = burst_minutes(x)
    hour = np.arange(MINUTES) // 60
    night = (hour >= NIGHT[0]) | (hour < NIGHT[1])
    out = []
    for keep in (np.ones(MINUTES, bool), night):
        n = valid[:, keep].sum(axis=1)
        rate = np.where(n > 0, burst[:, keep].sum(axis=1) / np.maximum(n, 1) * MINUTES, np.nan)
        out.append(np.where(ok, rate, np.nan))
    return out


def features(dates, rate):
    """The four columns for every day that has a rate; dates are datetime.date."""
    of = {d: r for d, r in zip(dates, rate) if np.isfinite(r)}
    def mean(day, first, last):
        v = [of[x] for k in range(first, last + 1) if (x := day - dt.timedelta(days=k)) in of]
        return np.mean(v) if v else np.nan
    rows = {}
    for day in of:
        base = mean(day, 14, 27)
        rows[day] = (np.log1p(of[day]), np.log1p(mean(day, 0, 6)), np.log1p(mean(day, 0, 27)),
                     np.log1p(mean(day, 0, 13)) - np.log1p(base) if np.isfinite(base) else 0.0)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("minutes", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    rows = []
    for f in sorted(args.minutes.glob("*.npz")):
        z = np.load(f)
        dates = [dt.date.fromisoformat(str(d)) for d in z["dates"]]
        every, night = day_rates(z["x"])
        a, b = features(dates, every), features(dates, night)
        for day in sorted(a):
            if day in b:
                rows.append((f.stem, day.isoformat(), *a[day], *b[day]))
        print(f"{f.stem}: {len(a)} days, median {np.nanmedian(every):.0f} bursts a day, {np.nanmedian(night):.0f} at "
              f"the night rate", flush=True)
    if not rows:
        sys.exit("no day with a rate: nothing written")
    cols = [f"{n}_{k}" for n in ("bursts", "night") for k in ("1", "7", "28", "rel")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["station", "date"] + cols).to_csv(args.out, index=False, lineterminator="\n")
    prov = args.minutes / "provenance.json"
    args.out.with_suffix(".provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(factor=FACTOR, bands=[BANDS[i] for i in BURST_BANDS], min_day=MIN_DAY, night_utc=NIGHT),
        inputs={str(prov): hashlib.sha256(prov.read_bytes()).hexdigest()} if prov.exists() else {},
        versions=dict(numpy=np.__version__, pandas=pd.__version__), rows=len(rows)), indent=1))
    print(f"{len(rows)} station-days -> {args.out}")


if __name__ == "__main__":
    main()
