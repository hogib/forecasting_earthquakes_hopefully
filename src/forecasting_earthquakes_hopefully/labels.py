"""The label of every station-day that has an envelope: the answer to the question of protocol.py.

    uv run python -m forecasting_earthquakes_hopefully.labels ENVELOPES_DIR OUT_DIR

For a station, the events of the question are those of catalog/events.csv with magnitude >= MIN_MAGNITUDE within
RADIUS_KM. One of them counts (is the first large event of a sequence) when no other lies within RADIUS_KM of
the station in the CLEAN_DAYS days before it. For a day d of the station:

  dropped   d is the day of an event of the question or one of the CLEAN_DAYS days after it: the recording holds
            the earthquake or its aftershocks, the day is neither trained on nor scored
  label 1   not dropped, and an event that counts falls on d+1 .. d+LEAD_DAYS
  label 0   not dropped otherwise

`group` names the sequence a day belongs to for leaving one out: the events of the question within RADIUS_KM of
any station of the region are chained in time, a gap above CLEAN_DAYS starts a new sequence, and a day has the
group of the sequence whose first event is nearest in time. Days are UTC. Held-out days have no envelope and
so no row.

  OUT_DIR/labels.csv        station, region, date, label, dropped, event_id, days_to_event, group
  OUT_DIR/sequences.csv     region, group, first and last event, events, stations with positive days
  OUT_DIR/provenance.json   command line, parameters, commit, hashes of the catalogue files, counts

Exits nonzero when no row is written.
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from . import protocol, split
from .survey import commit, km


def station_events(lat, lon, events):
    """The events of the question at a station, in time order: (time, id, magnitude, distance, counts)."""
    near = []
    for e in events.itertuples():
        d = km(lat, lon, e.latitude, e.longitude)
        if e.magnitude >= protocol.MIN_MAGNITUDE and d <= protocol.RADIUS_KM:
            near.append([e.time, e.id, e.magnitude, d])
    near.sort()
    for i, e in enumerate(near):
        e.append(i == 0 or (e[0] - near[i - 1][0]) > pd.Timedelta(days=protocol.CLEAN_DAYS))
    return near


def label_day(day, near):
    """(label, dropped, event id, days to the event) of one day given station_events()."""
    for t, *_ in near:
        if 0 <= (day - t.date()).days <= protocol.CLEAN_DAYS:
            return 0, True, "", ""
    for t, eid, _, _, counts in near:
        ahead = (t.date() - day).days
        if counts and 1 <= ahead <= protocol.LEAD_DAYS:
            return 1, False, eid, ahead
    return 0, False, "", ""


def sequences(region, stations, events):
    """The sequences of a region: lists of event times, chained while the gap is at most CLEAN_DAYS."""
    times = sorted({e[0] for s in stations.itertuples() if split.region_of(s.code) == region
                    for e in station_events(s.latitude, s.longitude, events)})
    out = []
    for t in times:
        if out and (t - out[-1][-1]) <= pd.Timedelta(days=protocol.CLEAN_DAYS):
            out[-1].append(t)
        else:
            out.append([t])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("envelopes", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    events = pd.read_csv(protocol.CATALOG / "events.csv", parse_dates=["time"])
    stations = pd.read_csv(protocol.CATALOG / "stations.csv").drop_duplicates("code")
    have = sorted(p.parent.name for p in args.envelopes.glob("*/index.csv"))
    stations = stations[stations["code"].isin(have) & ~stations["code"].isin(protocol.held_out_stations())]
    seqs = {r: sequences(r, stations, events) for r in sorted({split.region_of(c) for c in stations["code"]})}
    name = lambda r, s: f"{r}-{s[0]:%Y-%m}"
    rows = []
    for s in stations.itertuples():
        region, near = split.region_of(s.code), station_events(s.latitude, s.longitude, events)
        for day in pd.read_csv(args.envelopes / s.code / "index.csv")["date"]:
            day = dt.date.fromisoformat(day)
            if split.held_out(s.code, day):
                sys.exit(f"{s.code} {day}: a held-out day has an envelope")
            label, dropped, eid, ahead = label_day(day, near)
            seq = min(seqs[region], key=lambda q: abs((q[0].date() - day).days)) if seqs[region] else None
            rows.append(dict(station=s.code, region=region, date=day.isoformat(), label=label, dropped=int(dropped),
                             event_id=eid, days_to_event=ahead, group=name(region, seq) if seq else f"{region}-none"))
    if not rows:
        sys.exit("no station-day found: nothing written")
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "labels.csv", index=False, lineterminator="\n")
    pos = table[table["label"] == 1]
    pd.DataFrame([dict(region=r, group=name(r, q), first=q[0].isoformat(), last=q[-1].isoformat(), events=len(q),
                       positive_days=int((pos["group"] == name(r, q)).sum()),
                       stations=" ".join(sorted(pos[pos["group"] == name(r, q)]["station"].unique())))
                  for r, qs in seqs.items() for q in qs]).to_csv(args.out / "sequences.csv", index=False, lineterminator="\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (args.out / "provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        parameters=dict(min_magnitude=protocol.MIN_MAGNITUDE, radius_km=protocol.RADIUS_KM, lead_days=protocol.LEAD_DAYS,
                        clean_days=protocol.CLEAN_DAYS),
        inputs={f: sha(protocol.CATALOG / f) for f in ("events.csv", "stations.csv", "held_out_days_previous.csv")},
        versions=dict(pandas=pd.__version__), stations=have,
        rows=len(table), positive=int(table["label"].sum()), dropped=int(table["dropped"].sum())), indent=1))
    print(f"{len(table)} station-days, {int(table['label'].sum())} positive, {int(table['dropped'].sum())} dropped, "
          f"{table['station'].nunique()} stations -> {args.out}")


if __name__ == "__main__":
    main()
