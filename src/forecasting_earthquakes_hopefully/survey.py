"""Which days of which stations are on disk, and which earthquakes of the question lie near them.

    uv run python -m forecasting_earthquakes_hopefully.survey OUT_DIR ROOT [ROOT ...]

Every ROOT holds one folder per station with day files or archives of them, named
<NET>_<STATION>_<DDMMYYYY>_000000_<DDMMYYYY>_000000_<CHANNEL>.mseed (inside a .zip or not). Only the names are
read: no recording is opened, so a day counts as on disk when a file spans it, whatever gaps the file has.
The folders of the held-out stations (protocol.held_out_stations) are not entered.

Writes to OUT_DIR
  stations.csv    one line per station: channels, coordinates, days on disk, first and last day, roots
  stretches.csv   every unbroken stretch of days of every station
  events.csv      every earthquake of the question (protocol.py) within the radius of a station whose day, or
                  a day up to LEAD_DAYS before it, is on disk: distance, how many of the 2 * LEAD_DAYS days
                  before it are on disk, and whether it is clean (protocol.CLEAN_DAYS)
  provenance.json command line, parameters, commit, versions, sha256 of the catalogue files, files per root
Exits nonzero when no day is found.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from . import protocol

SPAN = re.compile(r"([A-Z0-9]+)_([A-Z0-9]+)_(\d{2})(\d{2})(\d{4})_000000_(\d{2})(\d{2})(\d{4})_000000_([A-Z]+)\.mseed")
KM_PER_DEG = 111.19


def km(lat, lon, lat2, lon2):
    return KM_PER_DEG * math.hypot(lat - lat2, (lon - lon2) * math.cos(math.radians(lat)))


def commit():
    root = Path(__file__).resolve().parent
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
        return rev + (" (uncommitted changes)" if dirty.strip() else "")
    except OSError:
        return "unknown"


def stretches(days):
    """(first, last) of every unbroken run in a sorted list of dates."""
    out, a = [], days[0]
    for x, y in zip(days, days[1:] + [None]):
        if y is None or (y - x).days > 1:
            out.append((a, x))
            a = y
    return out


def write(path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("roots", type=Path, nargs="+")
    args = ap.parse_args()
    held = protocol.held_out_stations()
    station_file, event_file = protocol.CATALOG / "stations.csv", protocol.CATALOG / "events.csv"
    where = {r["code"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(open(station_file))}
    quakes = [(dt.datetime.fromisoformat(r["time"]), float(r["magnitude"]), float(r["latitude"]), float(r["longitude"]),
               r["place"], r["id"]) for r in csv.DictReader(open(event_file))]
    quakes = sorted(q for q in quakes if q[1] >= protocol.MIN_MAGNITUDE)

    days, channels, roots = defaultdict(set), defaultdict(set), defaultdict(set)
    files, not_read, skipped = defaultdict(int), [], set()

    def add(name, root):
        m = SPAN.search(name)
        if not m:
            not_read.append(name)
            return
        station = m.group(2)
        if station in held:                       # a file of a held-out station in another station's folder
            skipped.add(station)
            return
        a = dt.date(int(m.group(5)), int(m.group(4)), int(m.group(3)))
        b = dt.date(int(m.group(8)), int(m.group(7)), int(m.group(6)))
        days[station].update(a + dt.timedelta(days=i) for i in range((b - a).days))
        channels[station].add(f"{m.group(1)} {m.group(9)}")
        roots[station].add(str(root))
        files[str(root)] += 1

    for root in args.roots:
        for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if folder.name in held:
                skipped.add(folder.name)
                continue
            for f in sorted(folder.rglob("*")):
                if f.suffix == ".zip":
                    try:
                        for name in zipfile.ZipFile(f).namelist():
                            add(name, root)
                    except zipfile.BadZipFile:
                        not_read.append(str(f))
                elif f.suffix == ".mseed":
                    add(f.name, root)
        print(f"{root}: {files[str(root)]} files", flush=True)
    if not days:
        sys.exit("no day found")

    lead = protocol.LEAD_DAYS
    station_rows, stretch_rows, event_rows = [], [], []
    for s in sorted(days):
        d = sorted(days[s])
        lat, lon = where.get(s, (None, None))
        runs = stretches(d)
        station_rows.append(dict(station=s, channels=", ".join(sorted(channels[s])), latitude=lat, longitude=lon, days=len(d),
                                 first=d[0], last=d[-1], stretches=len(runs),
                                 longest=max((b - a).days + 1 for a, b in runs), roots="; ".join(sorted(roots[s]))))
        stretch_rows += [dict(station=s, first=a, last=b, days=(b - a).days + 1) for a, b in runs]
        if lat is None:
            continue
        near = [q for q in quakes if km(lat, lon, q[2], q[3]) <= protocol.RADIUS_KM]
        for q in near:
            day = q[0].date()
            if not any(day - dt.timedelta(days=i) in days[s] for i in range(lead + 1)):
                continue
            clean = not any(dt.timedelta(0) < q[0] - o[0] <= dt.timedelta(days=protocol.CLEAN_DAYS) for o in near)
            event_rows.append(dict(station=s, time=q[0].isoformat(), magnitude=q[1], distance_km=round(km(lat, lon, q[2], q[3]), 1),
                                   days_before_on_disk=sum(day - dt.timedelta(days=i) in days[s] for i in range(1, 2 * lead + 1)),
                                   of_days=2 * lead, clean=clean, place=q[4], event_id=q[5]))

    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "stations.csv", station_rows)
    write(args.out / "stretches.csv", stretch_rows)
    if event_rows:
        write(args.out / "events.csv", event_rows)
    (args.out / "provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        parameters=dict(min_magnitude=protocol.MIN_MAGNITUDE, radius_km=protocol.RADIUS_KM, lead_days=lead,
                        clean_days=protocol.CLEAN_DAYS, roots=[str(r) for r in args.roots]),
        commit=commit(), versions=dict(python=platform.python_version()),
        input_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (station_file, event_file)},
        files_per_root=dict(files), held_out_stations_not_entered=sorted(skipped), names_not_understood=not_read,
        stations=len(station_rows), station_events=len(event_rows)), indent=1))
    print(f"DONE | {len(station_rows)} stations, {sum(r['days'] for r in station_rows)} station-days, "
          f"{len(event_rows)} station-events, {len(skipped)} held-out stations not entered, in {args.out}", flush=True)


if __name__ == "__main__":
    main()
