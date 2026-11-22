"""Changes of the seismic velocity around a station, day by day, from the correlations of correlations.py.

    uv run python -m forecasting_earthquakes_hopefully.velocity CORRELATIONS_DIR OUT.csv [--reference-until DATE]

The late part of a noise correlation is made of waves that wandered around the station. When the velocity of
the rock changes by a share dv/v, every one of them arrives earlier or later by the same share of its travel
time, so the late part is stretched or squeezed along the lag axis. Per station, band of BANDS and pair of
components:

  current     the mean of the correlations of the STACK_DAYS days ending on the day (days up to it only)
  reference   the mean over the days of the same era (a new era starts after ERA_GAP days without data: an
              instrument may have changed), or over those up to --reference-until
  dv/v        the stretch e that makes reference(lag * (1 + e)) most like current, searched over +-MAX_STRETCH
              on the lags from LAGS[0] to LAGS[1] seconds of both sides, with the correlation coefficient there

The pairs of a day are averaged with weights cc^2. The reference without --reference-until is taken from all
days of the era, later ones included: fine for looking at what an earthquake did, not for a forecast, where
it has to stop before the days that are scored.

  OUT.csv              station, date, band, dvv (percent, positive is faster), cc, days in the stack
  OUT.provenance.json  command line, parameters, commit, hash of every index read

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
from scipy.signal import butter, sosfiltfilt

from .correlations import FS, MAX_LAG, PAIRS
from .survey import commit

BANDS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]    # Hz
LAGS = (5.0, 25.0)          # s: after the direct waves, before the edge
STACK_DAYS = 5
MIN_STACK = 3
ERA_GAP = 180               # days
MAX_STRETCH = 0.03
STEPS = 241                 # stretches tried: steps of 0.025 percent
MIN_REFERENCE = 20          # days


def stretch(reference, current, lags):
    """(e, cc): the stretch of the lag axis with reference(lag * (1 + e)) closest to current on the lags kept
    (a boolean mask), refined between the steps by a parabola, and the correlation coefficient there."""
    t = (np.arange(len(reference)) - (len(reference) - 1) / 2) / FS
    tried = np.linspace(-MAX_STRETCH, MAX_STRETCH, STEPS)
    cur = current[lags] - current[lags].mean()
    cc = np.empty(STEPS)
    for i, e in enumerate(tried):
        ref = np.interp(t[lags] * (1 + e), t, reference)
        ref = ref - ref.mean()
        cc[i] = (ref * cur).sum() / max(np.sqrt((ref ** 2).sum() * (cur ** 2).sum()), 1e-30)
    k = int(np.argmax(cc))
    if 0 < k < STEPS - 1:
        a, b, c = cc[k - 1], cc[k], cc[k + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c != 0 else 0.0
        return float(tried[k] + shift * (tried[1] - tried[0])), float(b - 0.25 * (a - c) * shift)
    return float(tried[k]), float(cc[k])


def eras(dates):
    """Runs of dates (sorted datetime.date) without a hole longer than ERA_GAP days: lists of positions."""
    out = [[0]]
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days > ERA_GAP:
            out.append([])
        out[-1].append(i)
    return out


def station_velocity(dates, corr, until=None):
    """dates (sorted) and corr (n, pairs, lags) of one station -> rows (date, band, dvv percent, cc, days)."""
    t = (np.arange(corr.shape[2]) - (corr.shape[2] - 1) / 2) / FS
    keep = (np.abs(t) >= LAGS[0]) & (np.abs(t) <= LAGS[1])
    ordinal = np.array([d.toordinal() for d in dates])
    rows = []
    for low, high in BANDS:
        band = sosfiltfilt(butter(4, (low, high), btype="bandpass", fs=FS, output="sos"), corr, axis=2)
        for era in eras(dates):
            ref_days = [i for i in era if until is None or dates[i] <= until]
            if len(ref_days) < MIN_REFERENCE:
                continue
            reference = band[ref_days].mean(axis=0)
            for i in era:
                recent = [j for j in era if 0 <= ordinal[i] - ordinal[j] < STACK_DAYS]
                if len(recent) < MIN_STACK:
                    continue
                current = band[recent].mean(axis=0)
                found = [stretch(reference[p], current[p], keep) for p in range(len(PAIRS))]
                w = np.array([max(cc, 0) ** 2 for _, cc in found])
                if w.sum() == 0:
                    continue
                rows.append((dates[i].isoformat(), f"{low:g}-{high:g}", round(100 * float(np.dot(w, [e for e, _ in found]) / w.sum()), 4),
                             round(float(np.dot(w, [cc for _, cc in found]) / w.sum()), 3), len(recent)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("correlations", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--reference-until", type=dt.date.fromisoformat)
    args = ap.parse_args()
    rows, inputs = [], {}
    for index in sorted(args.correlations.glob("*/index.csv")):
        station = index.parent.name
        dates = sorted(dt.date.fromisoformat(d) for d in pd.read_csv(index)["date"])
        corr = np.stack([np.load(index.parent / f"{d.isoformat()}.npy") for d in dates])
        if corr.shape[2] != 2 * MAX_LAG * FS + 1:
            sys.exit(f"{station}: correlations of {corr.shape[2]} lags, not those of correlations.py")
        got = station_velocity(dates, corr, args.reference_until)
        rows += [(station, *r) for r in got]
        inputs[str(index)] = hashlib.sha256(index.read_bytes()).hexdigest()
        print(f"{station}: {len(dates)} days, {len(got)} values", flush=True)
    if not rows:
        sys.exit("no value: nothing written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["station", "date", "band", "dvv", "cc", "days"]).to_csv(args.out, index=False, lineterminator="\n")
    args.out.with_suffix(".provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(bands=BANDS, lags=LAGS, stack_days=STACK_DAYS, min_stack=MIN_STACK, era_gap=ERA_GAP,
                        max_stretch=MAX_STRETCH, steps=STEPS, min_reference=MIN_REFERENCE,
                        reference_until=args.reference_until.isoformat() if args.reference_until else None),
        inputs=inputs, versions=dict(numpy=np.__version__, pandas=pd.__version__), rows=len(rows)), indent=1))
    print(f"{len(rows)} values -> {args.out}")


if __name__ == "__main__":
    main()
