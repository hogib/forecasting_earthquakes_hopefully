"""How many days, sequences and station-event pairs each variant of the question leaves to train and score on.

    uv run python -m forecasting_earthquakes_hopefully.question_grid ENVELOPES_DIR MINUTES_DIR OUT.csv

For every combination of --magnitudes, --leads (days) and --radii (km) the labels are made as labels.py makes
them and counted on the station-days that are not dropped and have a usable window (windows.py). No model is
trained and no score is read: the table says which variants hold enough to be worth a run, nothing about how
well they can be forecast. Magnitudes below the smallest of catalog/events.csv cannot be counted from it.

  OUT.csv              one row per variant: scored days, positive days, positive share, dropped days, groups
                       among the scored days, groups with positive days, station-event pairs, positive days per region
  OUT.provenance.json  command line, values, commit, hashes of the catalogue files

Exits nonzero when no variant has a scored day.
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

from . import labels, protocol
from .survey import commit
from .windows import Windows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("envelopes", type=Path)
    ap.add_argument("minutes", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--magnitudes", type=float, nargs="+", default=[5.0, 5.5, 6.0])
    ap.add_argument("--leads", type=int, nargs="+", default=[7, 14, 28])
    ap.add_argument("--radii", type=float, nargs="+", default=[30.0, 50.0, 100.0])
    args = ap.parse_args()
    events = pd.read_csv(protocol.CATALOG / "events.csv", parse_dates=["time"])
    if min(args.magnitudes) < events["magnitude"].min():
        sys.exit(f"catalog/events.csv starts at magnitude {events['magnitude'].min()}: nothing written")
    days_of = labels.read_days(args.envelopes)
    stations = labels.read_stations(sorted(days_of))
    every, _ = labels.table(days_of, stations, events)
    w = Windows(args.minutes, every.assign(dropped=0))      # a window is usable or not whatever the question is
    usable = set(zip(w.table["station"], w.table["date"].astype(str)))
    print(f"{len(every)} station-days, {len(usable)} with a usable window", flush=True)
    rows = []
    for m, lead, radius in itertools.product(args.magnitudes, args.leads, args.radii):
        tab, seqs = labels.table(days_of, stations, events, m, radius, lead)
        has = pd.Series([k in usable for k in zip(tab["station"], tab["date"])], index=tab.index)
        ok = tab[(tab["dropped"] == 0) & has]
        pos = ok[ok["label"] == 1]
        row = dict(magnitude=m, lead_days=lead, radius_km=radius, days=len(ok), positive=len(pos),
                   share=round(len(pos) / len(ok), 4) if len(ok) else 0.0,
                   dropped=int(tab["dropped"].sum()), groups=ok["group"].nunique(),
                   groups_positive=pos["group"].nunique(), pairs=len(pos[["station", "event_id"]].drop_duplicates()),
                   stations_positive=pos["station"].nunique())
        row.update({f"positive_{r}": int((pos["region"] == r).sum()) for r in sorted(seqs)})
        rows.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    out = pd.DataFrame(rows)
    if not out["days"].sum():
        sys.exit("no variant has a scored day: nothing written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args.out.with_suffix(".provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(magnitudes=args.magnitudes, leads=args.leads, radii=args.radii, clean_days=protocol.CLEAN_DAYS),
        inputs={f: sha(protocol.CATALOG / f) for f in ("events.csv", "stations.csv", "held_out_days_previous.csv")},
        versions=dict(pandas=pd.__version__), station_days=len(every), usable_windows=len(usable), skipped=w.skipped),
        indent=1))
    print(f"{len(out)} variants -> {args.out}")


if __name__ == "__main__":
    main()
