"""The input of the classifier: for a station and a day, the DAYS days that end on that day.

Every window is put on its own scale: per channel the median and the spread (1.4826 * MAD, at least
MIN_SPREAD) of the first BASE_DAYS days of the window, and every value becomes (value - median) / spread,
cut at +-CLIP. So the model sees how the last two weeks differ from the two weeks before them, and nothing
that tells one station or one instrument from another by its level. No day outside the window is used, so
nothing of a left-out sequence leaks in through the scale.

A day counts as present when at least MIN_DAY of its minutes have every chosen component. A window is used
when its last day is present, at least MIN_BASE of the base days and at least MIN_PRESENT days in all.
Held-out days have no summary and are simply missing.

    uv run python -m forecasting_earthquakes_hopefully.windows MINUTES_DIR LABELS_CSV OUT_CSV

writes the loudness features for rivals.py --features: per band, `loud_<band>` the mean of the scaled values
(mean channel, all components) over the days after the base days: how much louder the station has become;
and `level_<band>` the median of the base days in log10 counts: how loud the station is. Exits nonzero
when no window is usable.
"""
import argparse
import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .envelopes import BANDS
from .minutes import COMPONENTS, MINUTES

DAYS = 28
BASE_DAYS = 14
MIN_BASE = 5
MIN_PRESENT = 14
MIN_DAY = 0.5
MIN_SPREAD = 0.05     # log10 units
CLIP = 8.0
NB = len(BANDS)


class Windows:
    def __init__(self, minutes_dir, labels, components=COMPONENTS):
        """labels: the table of labels.csv. self.table: its rows that are not dropped and have a usable window."""
        comp = [COMPONENTS.index(c) for c in components]
        self.channels = [s * len(COMPONENTS) * NB + c * NB + b for s in (0, 1) for c in comp for b in range(NB)]
        self.features, self.components = len(self.channels), len(comp)
        self.comp_of = np.array([k for _ in (0, 1) for k in range(len(comp)) for _ in range(NB)])
        self.x, rows, self.skipped = {}, [], dict(no_summary=0, last_day_missing=0, base_too_short=0, too_few_days=0)
        labels = labels[labels["dropped"] == 0]
        for station, part in labels.groupby("station"):
            f = Path(minutes_dir) / f"{station}.npz"
            if not f.exists():
                self.skipped["no_summary"] += len(part)
                continue
            z = np.load(f)
            x = z["x"][:, self.channels]
            valid = np.isfinite(x[:, :len(comp) * NB]).all(axis=1)                     # (n, 1440): every component
            present = valid.mean(axis=1) >= MIN_DAY
            row_of = {dt.date.fromisoformat(str(d)): i for i, d in enumerate(z["dates"]) if present[i]}
            self.x[station] = x
            for r in part.itertuples():
                day = r.date if isinstance(r.date, dt.date) else dt.date.fromisoformat(r.date)
                idx = np.array([row_of.get(day - dt.timedelta(days=DAYS - 1 - k), -1) for k in range(DAYS)])
                why = ("last_day_missing" if idx[-1] < 0 else "base_too_short" if (idx[:BASE_DAYS] >= 0).sum() < MIN_BASE
                       else "too_few_days" if (idx >= 0).sum() < MIN_PRESENT else None)
                if why:
                    self.skipped[why] += 1
                    continue
                base = x[idx[:BASE_DAYS][idx[:BASE_DAYS] >= 0]]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(base, axis=(0, 2))
                    mad = 1.4826 * np.nanmedian(np.abs(base - med[None, :, None]), axis=(0, 2))
                if not np.isfinite(med).all():
                    self.skipped["base_too_short"] += 1
                    continue
                rows.append((r.Index, idx, med.astype(np.float32), np.maximum(mad, MIN_SPREAD).astype(np.float32)))
        self.table = labels.loc[[r[0] for r in rows]].reset_index(drop=True)
        self.idx = [r[1] for r in rows]
        self.med = np.stack([r[2] for r in rows]) if rows else np.zeros((0, self.features), np.float32)
        self.spread = np.stack([r[3] for r in rows]) if rows else np.zeros((0, self.features), np.float32)

    def __len__(self):
        return len(self.table)

    def window(self, i):
        """z float32 (DAYS, features, 1440), 0 where there is nothing; mask float32 (DAYS, components, 1440);
        present bool (DAYS,)."""
        idx = self.idx[i]
        x = self.x[self.table.at[i, "station"]][np.maximum(idx, 0)]
        z = np.clip((x - self.med[i][None, :, None]) / self.spread[i][None, :, None], -CLIP, CLIP)
        present = idx >= 0
        ok = np.isfinite(z) & present[:, None, None]
        mask = np.stack([ok[:, :self.components * NB][:, self.comp_of[:self.components * NB] == k].all(axis=1)
                         for k in range(self.components)], axis=1)
        valid = mask[:, self.comp_of]
        return np.where(valid, z, 0.0).astype(np.float32), mask.astype(np.float32), present

    def loudness(self):
        """The loudness features of every window (see the module text)."""
        names, rows = [f"{a:g}-{b:g}" for a, b in BANDS], []
        for i in range(len(self)):
            z, mask, _ = self.window(i)
            mean = z[BASE_DAYS:, :self.components * NB].reshape(DAYS - BASE_DAYS, self.components, NB, MINUTES)
            w = mask[BASE_DAYS:, :, None, :]
            loud = (mean * w).sum(axis=(0, 1, 3)) / np.maximum(w.sum(axis=(0, 1, 3)), 1)
            level = self.med[i][:self.components * NB].reshape(self.components, NB).mean(axis=0)
            rows.append(dict(station=self.table.at[i, "station"], date=self.table.at[i, "date"],
                             **{f"loud_{n}": float(v) for n, v in zip(names, loud)},
                             **{f"level_{n}": float(v) for n, v in zip(names, level)}))
        return pd.DataFrame(rows)


def read_labels(path):
    t = pd.read_csv(path, keep_default_na=False)
    t["date"] = pd.to_datetime(t["date"]).dt.date
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("minutes", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    w = Windows(args.minutes, read_labels(args.labels))
    if not len(w):
        sys.exit(f"no usable window ({w.skipped}): nothing written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    w.loudness().to_csv(args.out, index=False, lineterminator="\n")
    print(f"{len(w)} windows, {int(w.table['label'].sum())} positive, skipped {w.skipped} -> {args.out}")


if __name__ == "__main__":
    main()
