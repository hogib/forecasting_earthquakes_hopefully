"""Daily noise correlations between the components of one station: the waveform without its loudness.

    uv run python -u -m forecasting_earthquakes_hopefully.correlations OUT_DIR ROOT [ROOT ...] --station ELZG [--jobs 4]

The envelopes keep how loud the ground is and throw the waveform away. This keeps the other half: how the noise
on one component follows the noise on another, which depends on the rock around the station and not on how
strong the noise is. velocity.py turns its slow changes into changes of the seismic velocity.

The files are those envelopes.py reads (ROOT/<STATION>/, same names). For every day that is not held out and
has the three components, the day is cut into pieces of SEGMENT seconds; a piece with a missing sample on any
component is left out. Every piece of every component is demeaned, detrended, resampled to FS Hz, band-passed
to WHITEN, reduced to its sign (one bit: an earthquake in the piece counts no more than the noise) and whitened
(every frequency of WHITEN the same amplitude). The three pairs of components (PAIRS) are then correlated and
the pieces of the day averaged. Sign and whitening are what removes the loudness; the response of the
instrument is not removed, its phase is the same every day as long as the instrument is.

  OUT_DIR/<STATION>/<YYYY-MM-DD>.npy   float32 (3 pairs, 2 * MAX_LAG * FS + 1): the correlation at lags from
                                       -MAX_LAG to +MAX_LAG seconds; for the pair AB a positive lag means A
                                       follows B
  OUT_DIR/<STATION>/index.csv          date, pieces used, pieces of the day, input rate, file, member
  OUT_DIR/<STATION>/provenance.json    command line, parameters, commit, versions, source files with size

Held-out days are cut out of the file before anything is computed from them (split.held_out). Days already in
the index are kept unless --overwrite; the index is rewritten after every file, so a stopped run goes on from
there. Exits nonzero when nothing is written.
"""
import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

from . import split
from .envelopes import DAY, read_components, sources
from .survey import commit

FS = 20                     # Hz after resampling
SEGMENT = 1800              # s per piece
MAX_LAG = 30                # s kept on each side
WHITEN = (0.3, 8.0)         # Hz: above the ocean band, below 0.45 of FS
EDGE = 0.1                  # Hz over which the whitening rises and falls at the ends of WHITEN
MIN_PIECES = 8              # of the 48 pieces of a day
PAIRS = ("ZN", "ZE", "NE")
COMPONENTS = "ZNE"


def prepare(x, fs):
    """One piece of one component at its own rate -> its whitened spectrum at FS (rfft of SEGMENT * FS samples)."""
    ratio = Fraction(FS / fs).limit_denominator(1000)
    x = resample_poly(detrend(x.astype(np.float64) - x.mean()), ratio.numerator, ratio.denominator)
    x = sosfiltfilt(butter(4, WHITEN, btype="bandpass", fs=FS, output="sos"), x)
    spectrum = np.fft.rfft(np.sign(x))
    f = np.fft.rfftfreq(len(x), 1 / FS)
    rise = np.clip((f - WHITEN[0]) / EDGE, 0, 1) * np.clip((WHITEN[1] - f) / EDGE, 0, 1)
    gain = np.sin(0.5 * np.pi * rise) ** 2
    return gain * spectrum / np.maximum(np.abs(spectrum), 1e-12)


def correlate(a, b, n):
    """Spectra from prepare() of n samples -> the correlation at lags -MAX_LAG..MAX_LAG s (a positive lag: a
    follows b), divided so that a component with itself gives 1 at lag 0."""
    c = np.fft.irfft(a * np.conj(b), n)
    norm = np.sqrt(np.fft.irfft(np.abs(a) ** 2, n)[0] * np.fft.irfft(np.abs(b) ** 2, n)[0])
    lag = MAX_LAG * FS
    return np.concatenate([c[-lag:], c[:lag + 1]]) / max(norm, 1e-12)


def day_correlations(signals, fs):
    """signals: {component: one day of samples at fs, NaN where missing} -> ((3, lags) float32 or None, pieces
    used, pieces of the day)."""
    per = int(round(SEGMENT * fs))
    total = len(signals["Z"]) // per
    out, used = np.zeros((len(PAIRS), 2 * MAX_LAG * FS + 1)), 0
    for k in range(total):
        part = {c: signals[c][k * per:(k + 1) * per] for c in COMPONENTS}
        if any(not np.isfinite(p).all() or p.std() == 0 for p in part.values()):
            continue
        spectra = {c: prepare(p, fs) for c, p in part.items()}
        out += np.stack([correlate(spectra[p[0]], spectra[p[1]], SEGMENT * FS) for p in PAIRS])
        used += 1
    return (out / used).astype(np.float32) if used >= MIN_PIECES else None, used, total


def one_file(job):
    """Correlates the wanted days of one file; returns their index rows, or an error text."""
    path, member, first, end, station, wanted, out = job
    try:
        raw = zipfile.ZipFile(path).read(member) if member else Path(path).read_bytes()
        signals, fs, _ = read_components(raw, first, end, COMPONENTS)
        del raw
        if any(signals[c] is None for c in COMPONENTS):
            return []                                   # a file without the three components gives no day
        per_day = int(round(DAY * fs))
        days = [first + dt.timedelta(days=i) for i in range((end - first).days)]
        rows = []
        for i, day in enumerate(days):
            if split.held_out(station, day) or day not in wanted:      # a held-out day is never touched
                continue
            c, used, total = day_correlations({k: signals[k][0][i * per_day:(i + 1) * per_day] for k in COMPONENTS}, fs)
            if c is None:
                continue
            tmp = out / f"{day.isoformat()}.tmp.npy"
            np.save(tmp, c)
            os.replace(tmp, out / f"{day.isoformat()}.npy")
            rows.append(dict(date=day.isoformat(), pieces=used, pieces_of_day=total, input_rate=fs,
                             file=Path(path).name, member=member or ""))
        return rows
    except Exception as error:      # one unreadable file must not stop the station
        return f"{type(error).__name__}: {error}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--station", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.out / args.station
    out.mkdir(parents=True, exist_ok=True)
    index_file = out / "index.csv"
    old = pd.read_csv(index_file).set_index("date").to_dict("index") if index_file.exists() and not args.overwrite else {}
    files = sources(args.roots, args.station)
    jobs, rows, seen, held, used = [], [], set(), 0, []
    for path, member, first, end in sorted(files, key=lambda f: (f[3] - f[2], f[2])):     # day files before long files
        wanted = set()
        for i in range((end - first).days):
            day = first + dt.timedelta(days=i)
            if day in seen:
                continue
            seen.add(day)
            if split.held_out(args.station, day):
                held += 1
            elif day.isoformat() in old and (out / f"{day.isoformat()}.npy").exists():
                rows.append(dict(date=day.isoformat(), **old[day.isoformat()]))
            else:
                wanted.add(day)
        if wanted:
            jobs.append((path, member, first, end, args.station, wanted, out))
            used.append(dict(file=path, member=member or "", bytes=os.path.getsize(path)))
    todo = sum(len(j[5]) for j in jobs)
    print(f"{args.station}: {len(files)} files, {len(seen)} days, {held} held out and not read, {len(rows)} already built, "
          f"{todo} to try from {len(jobs)} files", flush=True)

    def save_index(rows):
        if rows:
            index = pd.DataFrame(rows).sort_values("date")
            index.to_csv(index_file.with_suffix(".tmp"), index=False)
            os.replace(index_file.with_suffix(".tmp"), index_file)

    t0, failed, done = time.time(), [], 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for job, res in zip(jobs, pool.map(one_file, jobs)):
            if isinstance(res, str):
                failed.append((job[0], job[1] or "", res))
                print(f"  FAILED {Path(job[0]).name} {job[1] or ''}: {res}", flush=True)
            else:
                rows += res
                save_index(rows)
            done += len(job[5])
            rate = done / (time.time() - t0)
            print(f"  {done}/{todo} days tried | {len(rows)} written | {rate:.2f} days/s | "
                  f"ETA {(todo - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    if not rows:
        sys.exit(f"{args.station}: no day with the three components and {MIN_PIECES} whole pieces: nothing written")
    (out / "provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(fs=FS, segment=SEGMENT, max_lag=MAX_LAG, whiten=WHITEN, edge=EDGE, min_pieces=MIN_PIECES, pairs=PAIRS),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        days=len(rows), held_out_not_read=held, failed=failed, sources=used), indent=1))
    print(f"{args.station}: {len(rows)} days -> {out}, {len(failed)} files failed")


if __name__ == "__main__":
    main()
