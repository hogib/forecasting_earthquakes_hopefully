"""Earthquake catalogue around a point from the EMSC event service (FDSN): small events included.

The list of the question (M >= 5) is catalog/events.csv; this one is for looking at the small events.

    uv run python -m forecasting_earthquakes_hopefully.small_events OUT_CSV --lat 38.4978 --lon 38.9844 --start 2019-07-01 --end 2021-07-01

EMSC collects the solutions of the agencies (AFAD, KOERI, ...) and keeps one per event; the column `author`
says whose. The request holds only the centre, the radius and the dates. A period that returns as many events
as the limit of one request is split in two and asked again, so nothing is cut.

Writes OUT_CSV (time UTC, latitude, longitude, depth_km, magnitude, mag_type, author, event_id, place,
distance_km from the centre), sorted in time, and OUT_CSV with the suffix .provenance.json (command line,
parameters, commit, versions, every request with the sha256 of its answer). Exits nonzero with no event.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .survey import commit

URL = "https://www.seismicportal.eu/fdsnws/event/1/query"
LIMIT = 5000
KM_PER_DEG = 111.19


def ask(start, end, args, log):
    """The events of [start, end) as a table; splits the period when the answer is full."""
    q = dict(starttime=start.isoformat(), endtime=end.isoformat(), lat=args.lat, lon=args.lon,
             maxradius=round(args.radius_km / KM_PER_DEG, 4), format="text", limit=LIMIT)
    url = URL + "?" + urllib.parse.urlencode(q)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=120) as r:
                raw = r.read() if r.status == 200 else b""          # 204: no event
            break
        except urllib.error.URLError as error:
            if attempt == 3:
                raise
            print(f"  retry after: {error}", flush=True)
            time.sleep(10 * (attempt + 1))
    t = pd.read_csv(io.BytesIO(raw), sep="|") if raw.strip() else pd.DataFrame()
    log.append(dict(url=url, events=len(t), sha256=hashlib.sha256(raw).hexdigest()))
    print(f"{start} to {end}: {len(t)} events", flush=True)
    time.sleep(1)
    if len(t) < LIMIT:
        return [t] if len(t) else []
    middle = start + (end - start) / 2
    return ask(start, middle, args, log) + ask(middle, end, args, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius-km", type=float, default=100.0)
    ap.add_argument("--start", type=dt.datetime.fromisoformat, required=True)
    ap.add_argument("--end", type=dt.datetime.fromisoformat, required=True)
    ap.add_argument("--chunk-days", type=int, default=30)
    args = ap.parse_args()
    log, parts, a = [], [], args.start
    while a < args.end:
        b = min(a + dt.timedelta(days=args.chunk_days), args.end)
        parts += ask(a, b, args, log)
        a = b
    if not parts:
        sys.exit("no event in the catalogue for this request")
    t = pd.concat(parts, ignore_index=True)
    t.columns = [c.strip("#") for c in t.columns]
    t = t.rename(columns={"Time": "time", "Latitude": "latitude", "Longitude": "longitude", "Depth/km": "depth_km",
                          "Magnitude": "magnitude", "MagType": "mag_type", "Author": "author", "EventID": "event_id",
                          "EventLocationName": "place"})
    t = t.drop_duplicates("event_id")
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601").dt.tz_localize(None)
    lat0 = np.radians(args.lat)
    t["distance_km"] = (KM_PER_DEG * np.hypot(t["latitude"] - args.lat, (t["longitude"] - args.lon) * np.cos(lat0))).round(1)
    t = t.sort_values("time")[["time", "latitude", "longitude", "depth_km", "magnitude", "mag_type", "author", "event_id",
                               "place", "distance_km"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    t.to_csv(tmp, index=False)
    tmp.replace(args.out)
    args.out.with_suffix(".provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        parameters=dict(url=URL, lat=args.lat, lon=args.lon, radius_km=args.radius_km, start=args.start.isoformat(),
                        end=args.end.isoformat(), chunk_days=args.chunk_days, limit=LIMIT),
        commit=commit(), versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        events=len(t), output_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(), input_hashes=log), indent=1))
    print(f"DONE | {len(t)} events, {t['time'].min()} to {t['time'].max()}, in {args.out}", flush=True)


if __name__ == "__main__":
    main()
