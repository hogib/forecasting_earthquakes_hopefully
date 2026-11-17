"""A look at the band envelopes of one station before an earthquake, without a model.

    uv run python -m forecasting_earthquakes_hopefully.look ENVELOPES_DIR SMALL_EVENTS_CSV OUT_DIR --station ELZG \
        --start 2019-07-24 --event 2020-01-24T17:55:14 [--after-days 21] [--radius-km 30]

Reads ENVELOPES_DIR/<STATION>/<date>.npy (envelopes.py; held-out days are not there) and the catalogue of
small_events.py, whose events on held-out days are left out too. Writes to OUT_DIR

  <STATION>_days.csv       per day and band: the 10th, 50th and 99th percentile of log10 of the envelope over the
                           seconds of the day, and the median over the night hours (NIGHT, UTC)
  <STATION>_timeline.png   those numbers from --start to --after-days after the event; below them the catalogue
                           events within --radius-km of the station
  <STATION>_last30.png     the last 30 days before the event, per hour: the median of log10 of the envelope
"""
import argparse
import datetime as dt
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import protocol, split
from .envelopes import BANDS

NIGHT = (21, 2)        # UTC hours, from 21:00 to 02:00 (00:00 to 05:00 in Turkey)
KM_PER_DEG = 111.19


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("envelopes", type=Path)
    ap.add_argument("small_events", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--station", required=True)
    ap.add_argument("--start", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--event", type=dt.datetime.fromisoformat, required=True)
    ap.add_argument("--after-days", type=int, default=21)
    ap.add_argument("--radius-km", type=float, default=30.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = args.station
    end = args.event.date() + dt.timedelta(days=args.after_days)
    days = [args.start + dt.timedelta(days=i) for i in range((end - args.start).days + 1)]
    names = [f"{a:g}-{b:g} Hz" for a, b in BANDS]
    hour = np.arange(86400) // 3600
    night = (hour >= NIGHT[0]) | (hour < NIGHT[1])

    rows, hourly = [], {}
    for d in days:
        f = args.envelopes / s / f"{d}.npy"
        if split.held_out(s, d) or not f.exists():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log10(np.load(f))
        x[~np.isfinite(x)] = np.nan
        if np.isnan(x).all():
            continue
        for k, name in enumerate(names):
            v = x[k][np.isfinite(x[k])]
            n = x[k][night & np.isfinite(x[k])]
            if len(v) < 3600:
                continue
            rows.append(dict(date=d, band=name, p10=np.percentile(v, 10), p50=np.percentile(v, 50), p99=np.percentile(v, 99),
                             night=np.median(n) if len(n) >= 600 else np.nan, seconds=len(v)))
        if 0 <= (args.event.date() - d).days <= 30:
            with np.errstate(all="ignore"):
                hourly[d] = np.nanmedian(x.reshape(len(BANDS), 24, 3600), axis=2)
    t = pd.DataFrame(rows)
    t.to_csv(args.out / f"{s}_days.csv", index=False)

    where = pd.read_csv(protocol.CATALOG / "stations.csv").set_index("code").loc[s]
    c = pd.read_csv(args.small_events, parse_dates=["time"])
    c["km"] = KM_PER_DEG * np.hypot(c["latitude"] - where["latitude"],
                                    (c["longitude"] - where["longitude"]) * math.cos(math.radians(where["latitude"])))
    c = c[(c["km"] <= args.radius_km) & (c["time"].dt.date >= args.start) & (c["time"].dt.date <= end)]
    c = c[[not split.held_out(s, d) for d in c["time"].dt.date]]
    held = [d for d in days if split.held_out(s, d)]

    def shade(ax):
        for d in held:
            ax.axvspan(d, d + dt.timedelta(days=1), color="0.9", lw=0, zorder=0)
        ax.axvline(args.event, color="#c0392b", lw=1.0)

    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(15, 13), sharex=True)
    for ax, name in zip(axes, names):
        g = t[t["band"] == name].set_index("date").reindex(days)
        ax.plot(days, g["p99"], color="#c0392b", lw=0.7, label="99th percentile of the seconds (transients)")
        ax.plot(days, g["p50"], color="#1f4e79", lw=0.9, label="median")
        ax.plot(days, g["night"], color="#2e8b57", lw=0.9, label="median of the night hours")
        ax.plot(days, g["p10"], color="0.4", lw=0.7, label="10th percentile (floor)")
        ax.set_ylabel(f"{name}\nlog10 counts", fontsize=8)
        shade(ax)
    axes[0].legend(fontsize=7, ncol=4, loc="upper left")
    ax = axes[-1]
    ax.scatter(c["time"], c["magnitude"], s=4 + 3 * c["magnitude"] ** 2, c=c["km"], cmap="viridis_r", vmin=0, vmax=args.radius_km)
    ax.set_ylabel(f"catalogue events within\n{args.radius_km:g} km (magnitude;\ndark = near)", fontsize=8)
    shade(ax)
    axes[0].set_title(f"{s}: band envelopes per day and the catalogue. Red line: {args.event:%d %b %Y %H:%M} UTC. "
                      "Grey: held-out days, not read.", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(args.out / f"{s}_timeline.png", dpi=110)
    plt.close(fig)

    last = sorted(hourly)
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(15, 13), sharex=True)
    first = dt.datetime.combine(args.event.date() - dt.timedelta(days=30), dt.time())
    for k, (ax, name) in enumerate(zip(axes, names)):
        when = [dt.datetime.combine(d, dt.time(h)) for d in last for h in range(24)]
        ax.plot(when, np.concatenate([hourly[d][k] for d in last]), color="#1f4e79", lw=0.7)
        ax.set_ylabel(f"{name}\nlog10 counts", fontsize=8)
        ax.axvline(args.event, color="#c0392b", lw=1.0)
    ax = axes[-1]
    cc = c[(c["time"] >= first) & (c["time"] <= args.event + dt.timedelta(hours=6))]
    ax.scatter(cc["time"], cc["magnitude"], s=6 + 4 * cc["magnitude"] ** 2, c=cc["km"], cmap="viridis_r", vmin=0, vmax=args.radius_km)
    ax.axvline(args.event, color="#c0392b", lw=1.0)
    ax.set_ylabel("catalogue events\n(magnitude)", fontsize=8)
    ax.set_xlim(first, args.event + dt.timedelta(hours=6))
    axes[0].set_title(f"{s}: the last 30 days, hourly median of each band", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(args.out / f"{s}_last30.png", dpi=110)
    plt.close(fig)

    before = c[c["time"] < args.event]
    print(f"{len(t) // len(names)} days read, {len(held)} held out in the period")
    print(f"catalogue within {args.radius_km:g} km before the event: {len(before)} events, magnitudes "
          f"{before['magnitude'].min()} to {before['magnitude'].max()}, median {before['magnitude'].median()}")
    month = before.assign(days_before=(args.event - before["time"]).dt.days)
    print("  in the last 30 days:", (month["days_before"] < 30).sum(), "| last 14:", (month["days_before"] < 14).sum(),
          "| last 3:", (month["days_before"] < 3).sum())
    print(month[month["days_before"] < 30][["time", "magnitude", "km", "depth_km", "author"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
