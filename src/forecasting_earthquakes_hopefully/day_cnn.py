"""The simple classifier: one day of one station in, one score out, one sequence left out at a time.

    uv run python -u -m forecasting_earthquakes_hopefully.day_cnn MINUTES_DIR LABELS_CSV OUT_DIR [--epochs 20] [--seed 0]
        [--days 1] [--band all] [--scale train] [--shape]

The input is the day itself as minutes.py wrote it: 30 channels (mean and maximum per minute, three components,
five bands, log10) by 1440 minutes, a coarse spectrogram. No window of earlier days and no scaling by the
station's own past: each channel is only centred and scaled with the mean and spread of the training days of
the fold, so the level of the station stays in the picture. Minutes without data are 0 after that and three
more channels say which component has data. A plain convolutional network over time, trained end to end on the
labels with the positive days weighted up; settings fixed on 2026-11-22 before the first run.

  --days 28   the input is the 28 days that end on the day (windows.py), as quarter hours: the mean of the means
              and the maximum of the maxima of every 15 minutes, 2688 steps. One example per station-day, so
              the window slides a day at a time; training windows of a station that end within EMBARGO_DAYS
              days of a left-out day of it are left out, as in train.py
  --band k    only band k of envelopes.BANDS (0 to 4): 6 channels and the 3 masks
  --shape     the loudness is taken out of every time step (shape()): what is left is how the energy is spread
              over bands and components and how far the peaks stand above the mean. With --scale own the
              scale is the median and spread of these values over the window's base days
  --scale own with --days 28: every channel on the scale of the window's own base days (windows.py), which
              removes the level of the station; the default `train` is the centring described above

The days are those windows.py finds usable, so the scores stand beside those of train.py and rivals.py on the
same days. Scored as everything else (score.py): the days of the left-out group are never trained on.

  OUT_DIR/scores.csv, summary.csv, groups.csv, stations.csv, run.json, DONE    as train.py writes them

Every LOG_EVERY steps a line with the loss, the learning rate, the speed and the time left of the fold.
"""
import argparse
import datetime as dt
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import score
from .envelopes import BANDS
from .minutes import COMPONENTS
from .survey import commit
from .train import embargoed
from .windows import CLIP, MIN_SPREAD, Windows, read_labels

BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
DROPOUT = 0.3
LOG_EVERY = 50
QUARTER = 15                # minutes per step with --days 28
NB, NC = len(BANDS), len(COMPONENTS)


class DayNet(nn.Module):
    def __init__(self, channels, deep=False):
        super().__init__()
        def block(a, b, k, s):
            return [nn.Conv1d(a, b, k, stride=s, padding=k // 2), nn.GELU(), nn.GroupNorm(8, b)]
        more = block(96, 96, 3, 4) if deep else []           # a month: one more step down, days come into view
        self.body = nn.Sequential(*block(channels, 32, 7, 2), *block(32, 64, 5, 3), *block(64, 64, 5, 4), *block(64, 96, 3, 2), *more)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(2 * 96, 1))

    def forward(self, x):
        """x (B, channels, steps) -> (B,) logits."""
        h = self.body(x)
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)).squeeze(-1)


def shape(x):
    """x (..., 30, T), log10 mean then log10 maximum per component and band -> the same without the loudness:
    every mean minus the mean of the 15 means of its time step (bands and components against each other),
    every maximum minus its own mean (how far the peak stands out)."""
    half = x.shape[-2] // 2
    mean, peak = x[..., :half, :], x[..., half:, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmean(mean, axis=-2, keepdims=True)
    return np.concatenate([mean - level, peak - mean], axis=-2)


def days(w, shaped=False):
    """(values (N, 30, 1440) with NaN, masks (N, 3, 1440)) of the last day of every usable window."""
    x = np.stack([w.x[s][i[-1]] for s, i in zip(w.table["station"], w.idx)])
    if shaped:
        x = shape(x)
    mask = np.isfinite(x[:, :NC * NB].reshape(len(x), NC, NB, -1)).all(axis=2)
    return x, mask.astype(np.float32)


def months(w, own, shaped=False):
    """(values (N, 30, days * 96) with NaN, masks (N, 3, days * 96)) of every usable window, as quarter hours;
    with own=True on the scale of the window's base days."""
    quarters = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # a quarter hour without data
        for s, x in w.x.items():
            r = x.reshape(len(x), 2, NC * NB, -1, QUARTER)
            quarters[s] = np.concatenate([np.nanmean(r[:, 0], axis=3), np.nanmax(r[:, 1], axis=3)], axis=1)
            if shaped:
                quarters[s] = shape(quarters[s])
    n, steps = len(w.idx[0]), next(iter(quarters.values())).shape[2]
    out = np.full((len(w.idx), 2 * NC * NB, n * steps), np.nan, dtype=np.float32)
    for k, (s, idx) in enumerate(zip(w.table["station"], w.idx)):
        for d, i in enumerate(idx):
            if i >= 0:
                out[k, :, d * steps:(d + 1) * steps] = quarters[s][i]
        if own and shaped:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                base = quarters[s][w.base[k]]
                med = np.nanmedian(base, axis=(0, 2))
                mad = 1.4826 * np.nanmedian(np.abs(base - med[None, :, None]), axis=(0, 2))
            out[k] = np.clip((out[k] - med[:, None]) / np.maximum(mad, MIN_SPREAD)[:, None], -CLIP, CLIP)
        elif own:
            out[k] = np.clip((out[k] - w.med[k][:, None]) / w.spread[k][:, None], -CLIP, CLIP)
    mask = np.isfinite(out[:, :NC * NB].reshape(len(out), NC, NB, -1)).all(axis=2)
    return out, mask.astype(np.float32)


class Fold:
    both = True

    def __init__(self, w, args):
        self.args, self.table, self.folds = args, w.table, {}
        self.x, self.mask = days(w, args.shape) if args.days == 1 else months(w, args.scale == "own", args.shape)
        if args.band != "all":
            self.x = self.x[:, [s * NC * NB + c * NB + int(args.band) for s in (0, 1) for c in range(NC)]]
        self.row = {k: i for i, k in enumerate(zip(w.table["station"], w.table["date"]))}

    def __call__(self, train, test):
        a, name, dev = self.args, test["group"].iloc[0], self.args.device
        seed = a.seed + sorted(self.table["group"].unique()).index(name)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        rows = lambda t: np.array([self.row[k] for k in zip(t["station"], t["date"])])
        r_all = rows(train)
        if a.days > 1:
            train = train[~embargoed(train, test)]
        r_train, r_test = rows(train), rows(test)
        if a.scale == "own":
            mu, sd = 0.0, 1.0
        else:
            mu = np.nanmean(self.x[r_train], axis=(0, 2), keepdims=True)
            sd = np.nanstd(self.x[r_train], axis=(0, 2), keepdims=True) + 1e-6

        def tensor(r):
            z = np.nan_to_num((self.x[r] - mu) / sd, nan=0.0).astype(np.float32)
            return torch.from_numpy(np.concatenate([z, self.mask[r]], axis=1)).to(dev)
        x_train, x_test = tensor(r_train), tensor(r_test)
        y = torch.tensor(train["label"].to_numpy(), dtype=torch.float32, device=dev)
        if y.sum() == 0:
            sys.exit(f"{name}: no positive day to train on")
        weight = (len(y) - y.sum()) / y.sum()
        net = DayNet(x_train.shape[1], deep=a.days > 1).to(dev)
        per_epoch = max(len(y) // BATCH, 1)
        steps = a.epochs * per_epoch
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
        t0, run, losses = time.time(), [], []
        net.train()
        for step in range(1, steps + 1):
            k = torch.from_numpy(rng.choice(len(y), BATCH)).to(dev)
            loss = nn.functional.binary_cross_entropy_with_logits(net(x_train[k]), y[k], pos_weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run.append(loss.item())
            if step % LOG_EVERY == 0 or step == steps:
                rate = step / (time.time() - t0)
                print(f"{name} | step {step}/{steps} | loss {np.mean(run[-LOG_EVERY:]):.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                      f"{rate:.1f} steps/s | ETA {(steps - step) / rate / 60:.1f} min", flush=True)
            if step % per_epoch == 0:
                losses.append(round(float(np.mean(run)), 5))
                run = []
        net.eval()
        with torch.no_grad():
            score_of = lambda x: torch.cat([net(x[i:i + 256]) for i in range(0, len(x), 256)]).cpu().numpy()
            s_train, s_test = score_of(x_train), score_of(x_test)
        self.folds[name] = dict(train=len(train), positive=int(y.sum().item()), test=len(test), loss=losses,
                                parameters=sum(p.numel() for p in net.parameters()))
        if a.days > 1:                                       # score.out_of_fold wants every training day, embargoed too
            del x_train
            with torch.no_grad():
                s_train = np.concatenate([score_of(tensor(r_all[i:i + 512])) for i in range(0, len(r_all), 512)])
        return s_train, s_test


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("minutes", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=1, choices=[1, 28])
    ap.add_argument("--band", default="all", choices=["all", "0", "1", "2", "3", "4"])
    ap.add_argument("--scale", default="train", choices=["train", "own"])
    ap.add_argument("--shape", action="store_true", help="take the loudness out of every time step")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    w = Windows(args.minutes, read_labels(args.labels))
    if args.shape and args.band != "all":
        sys.exit("--shape compares the bands with each other and needs --band all: nothing written")
    if args.scale == "own" and args.days == 1:
        sys.exit("--scale own needs --days 28: nothing written")
    if not len(w) or w.table["label"].sum() == 0:
        sys.exit(f"no usable day or no positive one (skipped {w.skipped}): nothing written")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(w)} days, {int(w.table['label'].sum())} positive, {w.table['station'].nunique()} stations, "
          f"{w.table['group'].nunique()} groups, device {args.device}", flush=True)
    fold = Fold(w, args)
    scored = score.out_of_fold(w.table, fold)
    scored.to_csv(args.out / "scores.tmp", index=False, lineterminator="\n")
    (args.out / "scores.tmp").replace(args.out / "scores.csv")
    summary = score.summary(scored)
    pd.DataFrame([summary]).to_csv(args.out / "summary.csv", index=False, lineterminator="\n")
    score.by_group(scored).to_csv(args.out / "groups.csv", index=False, lineterminator="\n")
    score.by_station(scored).to_csv(args.out / "stations.csv", index=False, lineterminator="\n")
    (args.out / "run.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        constants=dict(batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY, dropout=DROPOUT),
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        days=len(w), positive=int(w.table["label"].sum()), folds=fold.folds, summary=summary), indent=1))
    (args.out / "DONE").touch()
    print(pd.DataFrame([summary]).to_string(index=False))
    print(score.by_group(scored).to_string(index=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
