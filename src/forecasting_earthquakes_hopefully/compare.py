"""The stations of a region side by side before an earthquake: is a rise at one station there at the others too?

    uv run python -m forecasting_earthquakes_hopefully.compare ENVELOPES_DIR OUT_PNG --region elazig \
        --event 2020-01-24T17:55:14 --near ELZG,SVRC,FRT [--days 30] [--baseline-days 60]

Per band and station: the hourly median of log10 of the envelope over the last --days days, minus the median
of the same over the --baseline-days days before that, so every station starts from its own usual level.
Stations of --near are drawn in colour, the others of the region in grey. Held-out days have no envelope.
"""
import argparse
import datetime as dt
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import split
from .envelopes import BANDS

COLOURS = ["#c0392b", "#e08a00", "#1f4e79", "#2e8b57"]


def hourly(env_dir, station, day):
    f = env_dir / station / f"{day}.npy"
    if not f.exists():
        return np.full((len(BANDS), 24), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(np.load(f))
    x[~np.isfinite(x)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x.reshape(len(BANDS), 24, 3600), axis=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("envelopes", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--region", choices=sorted(split.REGIONS), required=True)
    ap.add_argument("--event", type=dt.datetime.fromisoformat, required=True)
    ap.add_argument("--near", required=True, help="comma-separated")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--baseline-days", type=int, default=60)
    args = ap.parse_args()
    near = args.near.split(",")
    stations = [s for s in split.REGIONS[args.region] if (args.envelopes / s).is_dir()]
    last = [args.event.date() - dt.timedelta(days=i) for i in range(args.days, -1, -1)]
    base = [last[0] - dt.timedelta(days=i) for i in range(1, args.baseline_days + 1)]
    when = [dt.datetime.combine(d, dt.time(h)) for d in last for h in range(24)]
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(15, 12), sharex=True)
    for s in stations:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            level = np.nanmedian(np.concatenate([hourly(args.envelopes, s, d) for d in base], axis=1), axis=1)
        if not np.isfinite(level).any():
            print(f"{s}: no day in the baseline, left out")
            continue
        x = np.concatenate([hourly(args.envelopes, s, d) for d in last], axis=1) - level[:, None]
        colour = COLOURS[near.index(s) % len(COLOURS)] if s in near else "0.6"
        for k, ax in enumerate(axes):
            ax.plot(when, x[k], color=colour, lw=1.0 if s in near else 0.6, zorder=3 if s in near else 1,
                    label=f"{s}{'' if s in near else ' (far)'}")
        print(f"{s}: {int(np.isfinite(x[2]).sum())} of {len(when)} hours")
    for (a, b), ax in zip(BANDS, axes):
        ax.axvline(args.event, color="k", lw=1.0)
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        ax.set_ylabel(f"{a:g}-{b:g} Hz\nlog10, above usual", fontsize=8)
        ax.set_ylim(-0.8, 1.6)
    axes[0].legend(fontsize=7, ncol=len(stations), loc="upper left")
    axes[0].set_xlim(when[0], args.event + dt.timedelta(hours=3))
    axes[0].set_title(f"{args.region}: hourly median of each band, relative to the station's own level in the "
                      f"{args.baseline_days} days before. Black line: {args.event:%d %b %Y %H:%M} UTC.", fontsize=10, loc="left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)


if __name__ == "__main__":
    main()
