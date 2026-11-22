"""The daily noise level of a distant station in one band, from the hourly spectra of EarthScope MUSTANG.

    uv run python -u -m forecasting_earthquakes_hopefully.far_station OUT.csv --start 2011-06-01 --end 2011-11-10
        [--start ... --end ...] [--target IU.ANTO.00.BHZ.M] [--low 0.1] [--high 0.5]

A check, not an input of the forecast: when the stations of a region get louder together below 0.5 Hz, is a
station hundreds of kilometres away louder on the same days? Then it is the sea and the weather. For every day
the mean over the hourly spectra of the mean power (dB) between --low and --high Hz; a day needs MIN_SPECTRA
of them. Days already in OUT.csv are not asked for again, so the same command goes on after a stop.

  OUT.csv              date, level_db, spectra
  OUT.provenance.json  command line, parameters, commit, the address asked, days

Exits nonzero when OUT.csv ends up without a day.
"""
import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .survey import commit

URL = "https://service.earthscope.org/mustang/noise-psd/1/query?target={target}&starttime={a}T00:00:00&endtime={b}T00:00:00&format=text"
CHUNK_DAYS = 10
MIN_SPECTRA = 12
TIMEOUT = 180


def parse(text, low, high):
    """{iso date: [mean dB of each spectrum that starts that day]}."""
    out, day, values = {}, None, []

    def close():
        if day and values:
            out.setdefault(day, []).append(float(np.mean(values)))
    for line in text.splitlines():
        if line.startswith("start:"):
            close()
            day, values = line.split()[1][:10], []
        elif line[:1].isdigit() and "," in line:
            f, p = line.split(",")
            if low <= float(f) <= high:
                values.append(float(p))
    close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", action="append", required=True)
    ap.add_argument("--end", action="append", required=True)
    ap.add_argument("--target", default="IU.ANTO.00.BHZ.M")
    ap.add_argument("--low", type=float, default=0.1)
    ap.add_argument("--high", type=float, default=0.5)
    args = ap.parse_args()
    have = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame(columns=["date", "level_db", "spectra"])
    asked_file = args.out.with_suffix(".asked")
    asked = set(asked_file.read_text().split()) if asked_file.exists() else set()
    rows = have.to_dict("records")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for start, end in zip(args.start, args.end):
        a, end = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        while a <= end:
            b = min(a + dt.timedelta(days=CHUNK_DAYS), end + dt.timedelta(days=1))
            if a.isoformat() not in asked:
                t0 = time.time()
                try:
                    with urllib.request.urlopen(URL.format(target=args.target, a=a, b=b), timeout=TIMEOUT) as r:
                        text = r.read().decode() if r.status == 200 else ""
                except urllib.error.HTTPError as e:
                    if e.code not in (204, 404):
                        sys.exit(f"{a}: HTTP {e.code}; run the same command again to go on")
                    text = ""
                except (urllib.error.URLError, TimeoutError) as e:
                    sys.exit(f"{a}: {e}; run the same command again to go on")
                got = parse(text, args.low, args.high)
                rows += [dict(date=d, level_db=round(float(np.mean(v)), 3), spectra=len(v)) for d, v in sorted(got.items())
                         if len(v) >= MIN_SPECTRA]
                pd.DataFrame(rows).drop_duplicates("date").sort_values("date").to_csv(args.out, index=False, lineterminator="\n")
                asked.add(a.isoformat())
                asked_file.write_text("\n".join(sorted(asked)))
                print(f"{a} .. {b}: {len(got)} days in {time.time() - t0:.0f} s, {len(rows)} in all", flush=True)
            a = b
    table = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame()
    if not len(table):
        sys.exit("no day: nothing written")
    args.out.with_suffix(".provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(target=args.target, low=args.low, high=args.high, min_spectra=MIN_SPECTRA, url=URL),
        versions=dict(numpy=np.__version__, pandas=pd.__version__), days=len(table)), indent=1))
    print(f"{len(table)} days -> {args.out}")


if __name__ == "__main__":
    main()
