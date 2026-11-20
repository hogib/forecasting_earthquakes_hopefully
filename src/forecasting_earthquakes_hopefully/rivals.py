"""The rivals: forecasts that never look at a recording. The classifier has to beat them.

    uv run python -m forecasting_earthquakes_hopefully.rivals LABELS_CSV SMALL_EVENTS_DIR OUT_DIR [--features CSV]

  chance       a random number per day (seed 0): what the scoring gives to knowing nothing
  day_of_year  logistic regression on the sine and cosine of the day of the year
  small_events logistic regression on log(1 + n), n the events of magnitude >= SMALL_MAGNITUDE within RADIUS_KM
               of the station on the day and the COUNT_DAYS - 1 days before it, from the EMSC lists in
               SMALL_EVENTS_DIR (scripts/fetch_small_events.sh). Events on held-out days are not counted; n is
               scaled up by the share of the window that is not held out.
  station      the share of positive days of the station among the training days: knowing only where you are
  --features   a table with station, date and feature columns (for instance the loudness per band); each
               column prefix before the first "_" becomes one more rival: logistic regression on its columns

Every rival is fitted on the other sequences and scored on the one left out (score.py).

  OUT_DIR/rivals.csv         one row per rival: the numbers of score.summary
  OUT_DIR/rivals_groups.csv  per rival and left-out sequence (alarm share 0.10)
  OUT_DIR/scores.csv         station, date, label, group and the out-of-fold score of every rival
  OUT_DIR/provenance.json
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol, score, split
from .survey import commit, km

SMALL_MAGNITUDE = 3.0
COUNT_DAYS = 28
RIDGE = 1e-2


def logistic(columns):
    """A forecast for score.out_of_fold: ridge logistic regression on the columns, standardised on the training days."""
    def forecast(train, test):
        mu, sd = train[columns].mean().to_numpy(), train[columns].std().to_numpy() + 1e-9
        X = np.c_[np.ones(len(train)), (train[columns].to_numpy() - mu) / sd]
        y, w = train["label"].to_numpy(dtype=float), np.zeros(len(columns) + 1)
        reg = RIDGE * np.eye(len(w))
        reg[0, 0] = 0
        for _ in range(50):                                  # Newton steps
            p = 1 / (1 + np.exp(-X @ w))
            step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) / len(y) + reg, X.T @ (p - y) / len(y) + reg @ w)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return np.c_[np.ones(len(test)), (test[columns].to_numpy() - mu) / sd] @ w
    return forecast


def station_rate(train, test):
    rate = train.groupby("station")["label"].mean()
    return test["station"].map(rate).fillna(train["label"].mean()).to_numpy()


def small_event_counts(table, folder, stations):
    """log(1 + n) per row of table, n as in the module text."""
    files = sorted(Path(folder).glob("*.csv"))
    ev = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in files]).drop_duplicates("event_id")
    ev = ev[ev["magnitude"] >= SMALL_MAGNITUDE]
    out = np.zeros(len(table))
    for s, rows in table.groupby("station"):
        lat, lon = stations.loc[s, "latitude"], stations.loc[s, "longitude"]
        d = np.array([km(lat, lon, a, b) for a, b in zip(ev["latitude"], ev["longitude"])])
        days = pd.Series(ev[d <= protocol.RADIUS_KM]["time"].dt.date)
        days = days[[not split.held_out(s, x) for x in days]]
        per_day = days.value_counts()
        for i, day in zip(rows.index, rows["date"]):
            window = [day - dt.timedelta(days=k) for k in range(COUNT_DAYS)]
            open_days = [x for x in window if not split.held_out(s, x)]
            n = sum(per_day.get(x, 0) for x in open_days)
            out[table.index.get_loc(i)] = np.log1p(n * COUNT_DAYS / max(len(open_days), 1))
    return out, files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", type=Path)
    ap.add_argument("small_events", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--features", type=Path)
    args = ap.parse_args()
    table = pd.read_csv(args.labels, keep_default_na=False)
    table["date"] = pd.to_datetime(table["date"]).dt.date
    stations = pd.read_csv(protocol.CATALOG / "stations.csv").drop_duplicates("code").set_index("code")
    doy = 2 * np.pi * np.array([d.timetuple().tm_yday for d in table["date"]]) / 365.25
    table["doy_sin"], table["doy_cos"] = np.sin(doy), np.cos(doy)
    table["small"], files = small_event_counts(table, args.small_events, stations)
    table["chance"] = np.random.default_rng(0).random(len(table))
    rivals = dict(chance=lambda train, test: test["chance"].to_numpy(), station=station_rate,
                  day_of_year=logistic(["doy_sin", "doy_cos"]), small_events=logistic(["small"]))
    inputs = [args.labels] + files
    if args.features:
        f = pd.read_csv(args.features)
        f["date"] = pd.to_datetime(f["date"]).dt.date
        names = [c for c in f.columns if c not in ("station", "date")]
        table = table.merge(f, on=["station", "date"], how="inner")
        for prefix in sorted({c.split("_")[0] for c in names}):
            rivals[prefix] = logistic([c for c in names if c.split("_")[0] == prefix])
        inputs.append(args.features)
    if not len(table) or table["label"].sum() == 0:
        sys.exit("no day or no positive day: nothing written")
    rows, groups, scores = [], [], table[table["dropped"] == 0][["station", "region", "date", "label", "group", "event_id"]].reset_index(drop=True)
    for name, forecast in rivals.items():
        scored = score.out_of_fold(table, forecast)
        rows.append(dict(rival=name, **score.summary(scored)))
        groups.append(score.by_group(scored).assign(rival=name))
        scores[name] = scored["score"].to_numpy()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "rivals.csv", index=False, lineterminator="\n")
    pd.concat(groups).to_csv(args.out / "rivals_groups.csv", index=False, lineterminator="\n")
    scores.to_csv(args.out / "scores.csv", index=False, lineterminator="\n")
    (args.out / "provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(small_magnitude=SMALL_MAGNITUDE, count_days=COUNT_DAYS, ridge=RIDGE, alarm_shares=score.ALARM_SHARES,
                        radius_km=protocol.RADIUS_KM),
        inputs={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
        versions=dict(numpy=np.__version__, pandas=pd.__version__), rows=len(scores)), indent=1))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
