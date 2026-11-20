"""Per-minute summaries of the three-component envelopes: what the classifier reads.

    uv run python -u -m forecasting_earthquakes_hopefully.minutes ENVELOPES_DIR OUT_DIR [--jobs 4]

ENVELOPES_DIR/<STATION>/<date>.npy is (3, bands, 86400) from envelopes.py --component ZNE. For every minute
with at least MIN_SECONDS seconds of data, per component and band: log10 of the mean and log10 of the maximum
of the 1 Hz envelope. A spike of a few seconds survives in the maximum.

  OUT_DIR/<STATION>.npz     dates (n,) as text and x float32 (n, 2 * 3 * bands, 1440): channel
                            stat * 15 + component * 5 + band, stat 0 the mean and 1 the maximum, components
                            in the order Z, N, E; NaN where the minute has too little data
  OUT_DIR/provenance.json   command line, parameters, commit, versions, days and index hash per station

Exits nonzero when no day is written.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .envelopes import BANDS
from .survey import commit

MINUTES = 1440
MIN_SECONDS = 30
COMPONENTS = "ZNE"
CHANNELS = 2 * len(COMPONENTS) * len(BANDS)


def summarise(e):
    """(3, bands, 86400) -> float32 (2 * 3 * bands, 1440)."""
    r = e.reshape(len(COMPONENTS), len(BANDS), MINUTES, 60)
    enough = np.isfinite(r).sum(axis=3) >= MIN_SECONDS
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)             # a minute without data
        out = np.log10(np.stack([np.nanmean(r, axis=3), np.nanmax(r, axis=3)]))
    out[:, ~enough] = np.nan
    out[~np.isfinite(out)] = np.nan                                 # a dead channel gives zero
    return out.reshape(CHANNELS, MINUTES).astype(np.float32)


def one_station(job):
    folder, out = job
    dates = list(pd.read_csv(folder / "index.csv")["date"])
    x = np.stack([summarise(np.load(folder / f"{d}.npy")) for d in dates])
    tmp = out / f"{folder.name}.tmp.npz"
    np.savez(tmp, dates=np.array(dates), x=x)
    os.replace(tmp, out / f"{folder.name}.npz")
    return folder.name, len(dates), hashlib.sha256((folder / "index.csv").read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("envelopes", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    folders = sorted(p.parent for p in args.envelopes.glob("*/index.csv"))
    if not folders:
        sys.exit(f"no station with an index in {args.envelopes}: nothing written")
    args.out.mkdir(parents=True, exist_ok=True)
    done = {}
    with ProcessPoolExecutor(args.jobs) as pool:
        for name, n, sha in pool.map(one_station, [(f, args.out) for f in folders]):
            done[name] = dict(days=n, index_sha256=sha)
            print(f"{name}: {n} days", flush=True)
    (args.out / "provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(min_seconds=MIN_SECONDS, components=COMPONENTS, bands=BANDS),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__), stations=done), indent=1))
    print(f"DONE, {sum(d['days'] for d in done.values())} days of {len(done)} stations in {args.out}", flush=True)


if __name__ == "__main__":
    main()
