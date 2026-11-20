"""The classifier of the question, one sequence left out at a time.

    uv run python -u -m forecasting_earthquakes_hopefully.train MINUTES_DIR LABELS_CSV OUT_DIR
        [--components ZNE] [--pretrain-epochs 10] [--epochs 30] [--only GROUP] [--seed 0]

For every group of labels.csv (score.py): the windows of the other groups are the training set, without those
of a station that end within EMBARGO_DAYS days of a left-out day of the same station (their days would overlap).
On the training set alone:

  1. the day encoder is pretrained without labels (encoder.Reconstructor) for --pretrain-epochs epochs;
     with 0 it stays as it was drawn at random, which is the comparison "does the pretraining matter"
  2. the encoder is frozen and turns every day of every window into a vector
  3. the classifier (a GRU over the days) is trained on the labels, the positive days weighted up to balance

then the windows of the left-out group are scored. Nothing is tuned on the left-out group: the settings
are the constants below and the defaults of the command line, fixed before the first run.

  OUT_DIR/scores.csv     station, region, date, label, group, event_id, score, alarm@a
  OUT_DIR/summary.csv    the numbers of score.summary;  OUT_DIR/groups.csv  per left-out group
  OUT_DIR/run.json       command line, settings, commit, versions, counts, losses per fold
  OUT_DIR/fold_<group>.pt   encoder and classifier of the fold;  OUT_DIR/DONE at the end

Every LOG_EVERY steps a line with the loss, the learning rate, the speed and the time left of the phase.
"""
import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import score
from .encoder import Classifier, DayEncoder, Reconstructor
from .survey import commit
from .windows import DAYS, Windows, read_labels

EMBARGO_DAYS = DAYS
WIDTH = 64
PRETRAIN_WINDOWS = 8        # windows per pretraining step
PRETRAIN_DAYS = 8           # days drawn from each of them
PRETRAIN_LR = 1e-3
HIDDEN_SHARE = 0.25
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
ENCODE_BATCH = 16           # windows at a time through the frozen encoder
LOG_EVERY = 50


def embargoed(train, test):
    """True for the training rows of a station that end within EMBARGO_DAYS days of a left-out day of it."""
    out = np.zeros(len(train), dtype=bool)
    ordinal = lambda col: np.array([d.toordinal() for d in col])
    for station, part in test.groupby("station"):
        days = np.sort(ordinal(part["date"]))
        sel = np.where(train["station"].to_numpy() == station)[0]
        if not len(sel):
            continue
        t = ordinal(train["date"].iloc[sel])
        i = np.clip(np.searchsorted(days, t), 0, len(days) - 1)
        near = np.minimum(np.abs(days[i] - t), np.abs(days[np.maximum(i - 1, 0)] - t))
        out[sel] = near < EMBARGO_DAYS
    return out


class Fold:
    """forecast(train, test) for score.out_of_fold; called once per left-out group."""
    both = True

    def __init__(self, windows, args, device, log):
        self.w, self.args, self.device, self.log, self.folds = windows, args, device, log, {}
        self.row = {(s, d): i for i, (s, d) in enumerate(zip(windows.table["station"], windows.table["date"]))}

    def batch(self, rows, days=None, rng=None):
        z, m, p = zip(*(self.w.window(i) for i in rows))
        z, m, p = np.stack(z), np.stack(m), np.stack(p)
        if days:                                             # pretraining: some present days of each window
            pick = [rng.choice(np.where(q)[0], days) for q in p]
            z = np.concatenate([a[k] for a, k in zip(z, pick)])
            m = np.concatenate([a[k] for a, k in zip(m, pick)])
            return torch.from_numpy(z).to(self.device), torch.from_numpy(m).to(self.device)
        return torch.from_numpy(z).to(self.device), torch.from_numpy(m).to(self.device), torch.from_numpy(p)

    def ticker(self, name, total):
        t0 = time.time()

        def tick(step, loss, lr):
            if step % LOG_EVERY == 0 or step == total:
                rate = step / (time.time() - t0)
                print(f"{name} | step {step}/{total} | loss {loss:.4f} | lr {lr:.2e} | {rate:.1f} steps/s | "
                      f"ETA {(total - step) / rate / 60:.1f} min", flush=True)
        return tick

    def __call__(self, train, test):
        a, name = self.args, test["group"].iloc[0]
        fit = train[~embargoed(train, test)]
        rows_fit = np.array([self.row[k] for k in zip(fit["station"], fit["date"])])
        rows_train = np.array([self.row[k] for k in zip(train["station"], train["date"])])
        rows_test = np.array([self.row[k] for k in zip(test["station"], test["date"])])
        seed = a.seed + sorted(self.w.table["group"].unique()).index(name)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        encoder = DayEncoder(self.w.features, self.w.components, WIDTH).to(self.device)
        info = dict(train=len(train), fitted_on=len(fit), positive=int(fit["label"].sum()), test=len(test), pretrain_loss=[], loss=[])

        steps = a.pretrain_epochs * max(len(rows_fit) // PRETRAIN_WINDOWS, 1)
        if steps:
            model = Reconstructor(encoder, self.w.features, self.w.comp_of, WIDTH).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, PRETRAIN_LR, total_steps=steps)
            tick, run = self.ticker(f"{name} pretrain", steps), []
            for step in range(1, steps + 1):
                z, m = self.batch(rng.choice(rows_fit, PRETRAIN_WINDOWS), PRETRAIN_DAYS, rng)
                loss = model.loss(z, m, HIDDEN_SHARE)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                run.append(loss.item())
                tick(step, float(np.mean(run[-LOG_EVERY:])), sched.get_last_lr()[0])
                if step % max(steps // a.pretrain_epochs, 1) == 0:
                    info["pretrain_loss"].append(round(float(np.mean(run)), 5))
                    run = []

        encoder.eval()
        every = np.unique(np.concatenate([rows_train, rows_test]))
        vec, pres = {}, {}
        t0 = time.time()
        with torch.no_grad():
            for k in range(0, len(every), ENCODE_BATCH):
                rows = every[k:k + ENCODE_BATCH]
                z, m, p = self.batch(rows)
                v = encoder(z.flatten(0, 1), m.flatten(0, 1)).reshape(len(rows), DAYS, -1).cpu()
                for i, r in enumerate(rows):
                    vec[r], pres[r] = v[i], p[i].float()
        print(f"{name} encoded {len(every)} windows in {time.time() - t0:.0f} s", flush=True)
        stack = lambda rows: (torch.stack([vec[r] for r in rows]).to(self.device), torch.stack([pres[r] for r in rows]).to(self.device))

        clf = Classifier(WIDTH).to(self.device)
        y = torch.tensor(fit["label"].to_numpy(), dtype=torch.float32, device=self.device)
        if y.sum() == 0:
            sys.exit(f"{name}: no positive window to train on")
        weight = (len(y) - y.sum()) / y.sum()
        v_fit, p_fit = stack(rows_fit)
        per_epoch = max(len(rows_fit) // BATCH, 1)
        steps = a.epochs * per_epoch
        opt = torch.optim.AdamW(clf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
        tick, run = self.ticker(f"{name} classifier", steps), []
        clf.train()
        for step in range(1, steps + 1):
            k = torch.from_numpy(rng.choice(len(rows_fit), BATCH)).to(self.device)
            loss = nn.functional.binary_cross_entropy_with_logits(clf(v_fit[k], p_fit[k]), y[k], pos_weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run.append(loss.item())
            tick(step, float(np.mean(run[-LOG_EVERY:])), sched.get_last_lr()[0])
            if step % per_epoch == 0:
                info["loss"].append(round(float(np.mean(run)), 5))
                run = []
        clf.eval()
        with torch.no_grad():
            s_train, s_test = clf(*stack(rows_train)).cpu().numpy(), clf(*stack(rows_test)).cpu().numpy()
        torch.save(dict(encoder=encoder.state_dict(), classifier=clf.state_dict()), self.args.out / f"fold_{name}.pt")
        self.folds[name] = info
        return s_train, s_test


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("minutes", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--components", default="ZNE", choices=["ZNE", "Z"])
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--only", help="leave out this group alone (a trial); the summary then covers that group only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    w = Windows(args.minutes, read_labels(args.labels), args.components)
    if not len(w) or w.table["label"].sum() == 0:
        sys.exit(f"no usable window or no positive one (skipped {w.skipped}): nothing written")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(w)} windows, {int(w.table['label'].sum())} positive, {w.table['station'].nunique()} stations, "
          f"{w.table['group'].nunique()} groups, skipped {w.skipped}, device {args.device}", flush=True)
    fold = Fold(w, args, args.device, print)
    table = w.table
    if args.only:
        if args.only not in set(table["group"]):
            sys.exit(f"no group {args.only}; there are {sorted(table['group'].unique())}")
        forecast = lambda train, test: fold(train, test) if test["group"].iloc[0] == args.only else (np.zeros(len(train)), np.full(len(test), np.nan))
        forecast.both = True
    else:
        forecast = fold
    scored = score.out_of_fold(table, forecast)
    if args.only:
        scored = scored[scored["group"] == args.only]
    scored.to_csv(args.out / "scores.tmp", index=False, lineterminator="\n")
    (args.out / "scores.tmp").replace(args.out / "scores.csv")
    summary = score.summary(scored)
    pd.DataFrame([summary]).to_csv(args.out / "summary.csv", index=False, lineterminator="\n")
    score.by_group(scored).to_csv(args.out / "groups.csv", index=False, lineterminator="\n")
    (args.out / "run.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), commit=commit(),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        constants=dict(embargo_days=EMBARGO_DAYS, width=WIDTH, pretrain_windows=PRETRAIN_WINDOWS, pretrain_days=PRETRAIN_DAYS,
                       pretrain_lr=PRETRAIN_LR, hidden_share=HIDDEN_SHARE, batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY),
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        windows=len(w), positive=int(w.table["label"].sum()), skipped=w.skipped, folds=fold.folds, summary=summary), indent=1))
    (args.out / "DONE").touch()
    print(pd.DataFrame([summary]).to_string(index=False))
    print(score.by_group(scored).to_string(index=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
