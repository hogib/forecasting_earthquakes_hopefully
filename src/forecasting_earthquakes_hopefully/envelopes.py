"""Band envelopes at 1 Hz: the input of the models. One array per station and day.

    uv run python -u -m forecasting_earthquakes_hopefully.envelopes OUT_DIR ROOT [ROOT ...] --station ELZG [--component Z] [--jobs 4]

Every ROOT/<STATION>/ holds MiniSEED files of the station, inside .zip archives or not, named
<NET>_<STATION>_<DDMMYYYY>_000000_<DDMMYYYY>_000000_<CHANNEL>.mseed, of one day or of several. Only the
channel of --component (Z, the vertical, unless given) is used, chosen by name, never by its place in the file.
Envelopes of different components belong in different OUT_DIRs.

For every day that is not held out (split.held_out):

  OUT_DIR/<STATION>/<YYYY-MM-DD>.npy   float32 (bands, 86400): for each band of BANDS the root mean square of
                                       the band-passed signal in every second of the day (UTC), in counts;
                                       NaN where the second has a missing sample
  OUT_DIR/<STATION>/index.csv          date, coverage (share of the seconds with a value in the lowest band),
                                       sampling rate, gaps, source file and member, channel
  OUT_DIR/<STATION>/provenance.json    command line, parameters, commit, versions, source files with size

Held-out days are cut out of the file before anything is computed from them. Gaps shorter than GAP_FILL
seconds are interpolated; a longer gap splits the signal into pieces that are demeaned, detrended and filtered
on their own (zero-phase Butterworth), and pieces shorter than MIN_PIECE seconds are dropped. A file of several
days is filtered across midnight, so only the ends of a file and of a gap carry the edge effect of the filter.
A band whose upper edge is not below 0.45 of the sampling rate is left NaN. The response of the instrument
is not removed. Days already in the index are kept unless --overwrite. Exits nonzero when nothing is written.
"""
import argparse
import datetime as dt
import io
import json
import os
import platform
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt

from . import split
from .survey import commit

BANDS = [(0.1, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0)]   # Hz
COMPONENT = "Z"      # the default of --component
FILTER_ORDER = 4
GAP_FILL = 2.0       # s
MIN_PIECE = 60.0     # s, six periods of the lowest frequency
DAY = 86400
SPAN = re.compile(r"([A-Z0-9]+)_([A-Z0-9]+)_(\d{2})(\d{2})(\d{4})_000000_(\d{2})(\d{2})(\d{4})_000000_([A-Z]+)\.mseed")


def span(name):
    """(station, first day, day after the last) from a file name, or None."""
    m = SPAN.search(name)
    if not m:
        return None
    return (m.group(2), dt.date(int(m.group(5)), int(m.group(4)), int(m.group(3))),
            dt.date(int(m.group(8)), int(m.group(7)), int(m.group(6))))


def sources(roots, station):
    """Every MiniSEED file of the station under the roots: (path, member in the zip or None, first, end)."""
    out = []
    for root in roots:
        folder = Path(root) / station
        if not folder.is_dir():
            continue
        for f in sorted(folder.rglob("*")):
            if f.suffix == ".zip":
                for name in zipfile.ZipFile(f).namelist():
                    s = span(name)
                    if s and s[0] == station:
                        out.append((str(f), name, s[1], s[2]))
            elif f.suffix == ".mseed":
                s = span(f.name)
                if s and s[0] == station:
                    out.append((str(f), None, s[1], s[2]))
    return out


def pieces(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def band_rms(x, fs):
    """x: samples on a regular grid, NaN where missing, a whole number of seconds long.
    Returns float32 (bands, seconds) and the number of gaps left in x."""
    x = x.astype(np.float64)
    n_fs = int(round(fs))
    if abs(fs - n_fs) > 1e-6 or len(x) % n_fs:
        raise ValueError(f"sampling rate {fs} Hz or length {len(x)} does not give whole seconds")
    valid = np.isfinite(x)
    for a, b in pieces(~valid):                       # short gaps inside the data: linear interpolation
        if a > 0 and b < len(x) and (b - a) < GAP_FILL * fs:
            x[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
    valid = np.isfinite(x)
    runs = [(a, b) for a, b in pieces(valid) if (b - a) >= MIN_PIECE * fs]
    out = np.full((len(BANDS), len(x) // n_fs), np.nan, dtype=np.float32)
    for k, (lo, hi) in enumerate(BANDS):
        if hi > 0.45 * fs:
            continue
        sos = butter(FILTER_ORDER, [lo, hi], btype="bandpass", fs=fs, output="sos")
        y = np.full(len(x), np.nan)
        for a, b in runs:
            y[a:b] = sosfiltfilt(sos, detrend(x[a:b] - x[a:b].mean(), type="linear"))
        with np.errstate(invalid="ignore"):
            out[k] = np.sqrt((y.reshape(-1, n_fs) ** 2).mean(axis=1))      # NaN when a sample of the second is missing
    return out, max(len(pieces(valid)) - 1, 0)


def read_component(raw, first, end, component=COMPONENT):
    """The samples of the component from first 00:00 to end 00:00 on a regular grid (NaN where no data),
    the sampling rate and the name of the channel."""
    from obspy import UTCDateTime, read
    st = [tr for tr in read(io.BytesIO(raw)) if tr.stats.channel.endswith(component)]
    if not st:
        raise ValueError(f"no {component} channel in the file")
    rates, names = {tr.stats.sampling_rate for tr in st}, {tr.stats.channel for tr in st}
    if len(rates) != 1 or len(names) != 1:
        raise ValueError(f"several sampling rates or {component} channels: {sorted(rates)}, {sorted(names)}")
    fs = rates.pop()
    n = int(round((end - first).days * DAY * fs))
    x = np.full(n, np.nan, dtype=np.float32)
    start = UTCDateTime(first.year, first.month, first.day)
    for tr in st:
        i = int(round((tr.stats.starttime - start) * fs))
        lo, hi = max(i, 0), min(i + tr.stats.npts, n)
        if hi > lo:
            x[lo:hi] = tr.data[lo - i:hi - i]
    return x, fs, names.pop()


def one_file(job):
    """Builds the days of one file that are wanted; returns their index rows, or an error text."""
    path, member, first, end, station, wanted, out, component = job
    try:
        raw = zipfile.ZipFile(path).read(member) if member else Path(path).read_bytes()
        x, fs, channel = read_component(raw, first, end, component)
        del raw
        per_day = int(round(DAY * fs))
        days = [first + dt.timedelta(days=i) for i in range((end - first).days)]
        for i, day in enumerate(days):                 # a held-out day is cut out before anything is computed
            if split.held_out(station, day):
                x[i * per_day:(i + 1) * per_day] = np.nan
        env, _ = band_rms(x, fs)
        rows = []
        for i, day in enumerate(days):
            if day not in wanted:
                continue
            e = env[:, i * DAY:(i + 1) * DAY]
            seconds = np.isfinite(x[i * per_day:(i + 1) * per_day]).reshape(DAY, -1).all(axis=1)
            tmp = out / f"{day.isoformat()}.tmp.npy"
            np.save(tmp, e)
            os.replace(tmp, out / f"{day.isoformat()}.npy")
            rows.append(dict(date=day.isoformat(), coverage=round(float(np.isfinite(e[0]).mean()), 5),
                             input_rate=fs, gaps=max(len(pieces(seconds)) - 1, 0), channel=channel,
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
    ap.add_argument("--component", choices=["Z", "N", "E"], default=COMPONENT)
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
            jobs.append((path, member, first, end, args.station, wanted, out, args.component))
            used.append(dict(file=path, member=member or "", bytes=os.path.getsize(path)))
    todo = sum(len(j[5]) for j in jobs)
    print(f"{args.station}: {len(files)} files, {len(seen)} days, {held} held out and not read, {len(rows)} already built, "
          f"{todo} to build from {len(jobs)} files", flush=True)

    t0, failed, done = time.time(), [], 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for job, res in zip(jobs, pool.map(one_file, jobs)):
            if isinstance(res, str):
                failed.append((job[0], job[1] or "", res))
                print(f"[FAILED] {Path(job[0]).name}: {res}", flush=True)
            else:
                rows += res
            done += len(job[5])
            rate = done / (time.time() - t0)
            print(f"{args.station} {done}/{todo} days | {rate:.2f} days/s | ETA {(todo - done) / rate / 60:.1f} min", flush=True)

    if not rows:
        sys.exit(f"{args.station}: nothing was written")
    index = pd.DataFrame(rows).sort_values("date")
    index.to_csv(index_file.with_suffix(".tmp"), index=False)
    os.replace(index_file.with_suffix(".tmp"), index_file)
    import obspy
    import scipy
    (out / "provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        parameters=dict(bands=BANDS, component=args.component, filter_order=FILTER_ORDER, gap_fill=GAP_FILL, min_piece=MIN_PIECE,
                        split=dict(anchor=split.ANCHOR.isoformat(), block_days=split.BLOCK_DAYS,
                                   test_share=split.TEST_SHARE, seed=split.SEED)),
        commit=commit(), versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                       obspy=obspy.__version__, pandas=pd.__version__),
        days=len(index), held_out_days_not_read=held, failed=failed, input_files=used), indent=1))
    print(f"{args.station}: DONE, {len(index)} days in {out}, {held} held out, {len(failed)} files failed, "
          f"mean coverage {index['coverage'].mean():.3f}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
