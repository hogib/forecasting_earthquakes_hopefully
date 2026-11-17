"""Trains the model of the normal days (model.py) for one region and one fold, and writes what it leaves over.

    uv run python -u -m forecasting_earthquakes_hopefully.normal ENVELOPES_DIR OUT_DIR --region elazig --fold 0

A sample is (day, station): the station's day is predicted from the other stations of the region that have
the day (at least MIN_SOURCES). No label is used. The days get a role from the calendar and the catalogue:

  dropped  the day of an earthquake of the question (protocol.py) within the radius of any station of the
           region, and the AFTER_DAYS days after it
  before   the BEFORE_DAYS days before such an earthquake: never trained on, always predicted
  val      the days of the 14-day blocks (the grid of split.py) of this fold: not trained on, predicted
  train    all other days
Held-out days have no envelope and so are in no sample.

Each band is put on the scale of its station first: (log10 of the envelope - median) / (1.4826 * MAD) over a
sample of the station's training days, clipped at +-CLIP.

Writes OUT_DIR/<region>_f<fold>/  model.pt, run.json, DONE
  surprise.csv   per predicted (day, station): role, the mean over the minutes of observed minus predicted for
                 the mean of each band, the 95th percentile of the same for the maximum of each band, minutes
  before.npz     observed minus predicted per minute for the `before` days (rows as in before.csv)
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import protocol, split
from .envelopes import BANDS
from .model import MINUTES, Normal
from .survey import commit, km

BEFORE_DAYS = 28
AFTER_DAYS = 30
MIN_SOURCES = 2
MIN_COVERAGE = 0.5
CLIP = 8.0
SCALE_DAYS = 40          # training days per station that the scales are taken from
LOG_EVERY = 25           # steps


def region_events(stations):
    """The days of the earthquakes of the question within the radius of any of the stations."""
    where = {r["code"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(open(protocol.CATALOG / "stations.csv"))}
    days = set()
    for r in csv.DictReader(open(protocol.CATALOG / "events.csv")):
        if float(r["magnitude"]) >= protocol.MIN_MAGNITUDE and any(
                km(*where[s], float(r["latitude"]), float(r["longitude"])) <= protocol.RADIUS_KM for s in stations if s in where):
            days.add(dt.datetime.fromisoformat(r["time"]).date())
    return sorted(days)


def role_of(day, events, region, fold, folds):
    if any(dt.timedelta(0) <= day - e <= dt.timedelta(days=AFTER_DAYS) for e in events):
        return None
    if any(dt.timedelta(0) < e - day <= dt.timedelta(days=BEFORE_DAYS) for e in events):
        return "before"
    block = (day - split.ANCHOR).days // split.BLOCK_DAYS
    h = int.from_bytes(hashlib.sha256(f"fold:{region}:{block}".encode()).digest()[:8], "big")
    return "val" if h % folds == fold else "train"


def samples(env_dir, region, fold, folds):
    """The stations of the region with envelopes, and the table of (date, target, role, sources)."""
    stations = [s for s in split.REGIONS[region] if (env_dir / s / "index.csv").exists()]
    have = {}
    for s in stations:
        t = pd.read_csv(env_dir / s / "index.csv")
        have[s] = {dt.date.fromisoformat(d) for d, c in zip(t["date"], t["coverage"]) if c >= MIN_COVERAGE}
    events = region_events(stations)
    rows = []
    for day in sorted(set().union(*have.values())):
        role = role_of(day, events, region, fold, folds)
        there = [s for s in stations if day in have[s]]
        if role is None or len(there) < MIN_SOURCES + 1:
            continue
        rows += [dict(date=day, station=s, role=role, sources=len(there) - 1) for s in there]
    return stations, pd.DataFrame(rows), events


def load_log(env_dir, station, day):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(np.load(env_dir / station / f"{day}.npy"))
    x[~np.isfinite(x)] = np.nan
    return x


def scales(env_dir, table, stations, seed):
    """Per station: median and 1.4826 * MAD of log10 of the envelope per band, from training days."""
    rng, out = np.random.default_rng(seed), {}
    for s in stations:
        days = table[(table["station"] == s) & (table["role"] == "train")]["date"].tolist()
        if not days:
            sys.exit(f"{s}: no training day")
        pick = rng.choice(len(days), min(SCALE_DAYS, len(days)), replace=False)
        x = np.concatenate([load_log(env_dir, s, days[i])[:, ::10] for i in pick], axis=1)
        med = np.nanmedian(x, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(x - med[:, None]), axis=1)
        out[s] = (med.astype(np.float32), np.maximum(mad, 1e-3).astype(np.float32))
    return out


class Days(torch.utils.data.Dataset):
    def __init__(self, env_dir, table, stations, scale, have):
        """table: the samples to predict; have: every (date, station) with a usable day, the sources come from it."""
        self.dir, self.table, self.stations, self.scale, self.have = env_dir, table.reset_index(drop=True), stations, scale, have

    def __len__(self):
        return len(self.table)

    def z(self, station, day):
        med, mad = self.scale[station]
        return np.clip((load_log(self.dir, station, day) - med[:, None]) / mad[:, None], -CLIP, CLIP)

    def __getitem__(self, i):
        r = self.table.iloc[i]
        S, nb = len(self.stations), len(BANDS)
        x = np.zeros((S, nb + 1, 86400), dtype=np.float32)
        present = np.zeros(S, dtype=bool)
        for k, s in enumerate(self.stations):
            if s == r["station"] or (r["date"], s) not in self.have:
                continue
            z = self.z(s, r["date"])
            ok = np.isfinite(z).all(axis=0)
            x[k, :nb], x[k, nb], present[k] = np.where(ok, z, 0.0), ok, True
        z = self.z(r["station"], r["date"]).reshape(nb, MINUTES, 60)
        enough = np.isfinite(z).sum(axis=2) >= 30
        with warnings.catch_warnings():                       # a minute without data: mean of an empty slice
            warnings.simplefilter("ignore", RuntimeWarning)
            y = np.concatenate([np.nanmean(z, axis=2), np.nanmax(z, axis=2)]).T              # (1440, 2 * bands)
        valid = np.concatenate([enough, enough]).T
        y = np.where(valid, y, 0.0).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(present), self.stations.index(r["station"]),
                r["date"].weekday(), torch.from_numpy(y), torch.from_numpy(valid), i)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("envelopes", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--region", choices=sorted(split.REGIONS), required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only this many training samples (for a trial)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stations, table, events = samples(args.envelopes, args.region, args.fold, args.folds)
    if table.empty or not (table["role"] == "train").any():
        sys.exit("no training sample")
    scale = scales(args.envelopes, table, stations, args.seed)
    train = table[table["role"] == "train"]
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=args.seed)
    rest = table[table["role"] != "train"]
    have = {(r.date, r.station) for r in table.itertuples()}
    out = args.out / f"{args.region}_f{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)
    print(f"{args.region} fold {args.fold}/{args.folds} on {device} | stations {stations} | events {[str(e) for e in events]}", flush=True)
    print("samples (day, station) by role: " + ", ".join(f"{k} {v}" for k, v in table["role"].value_counts().items())
          + f" | days {table['date'].min()} to {table['date'].max()}", flush=True)

    model = Normal(len(stations), len(BANDS), args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loader = torch.utils.data.DataLoader(Days(args.envelopes, train, stations, scale, have), batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    steps = args.epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    amp = torch.amp.GradScaler(enabled=device == "cuda")
    print(f"{n_params} parameters | {len(train)} training samples, {len(loader)} steps per epoch, {steps} steps", flush=True)

    def run(batch):
        x, present, target, weekday, y, valid, _ = batch
        with torch.autocast(device, enabled=device == "cuda"):
            p = model(x.to(device), present.to(device), target.to(device), weekday.to(device))
        return p.float(), y.to(device), valid.to(device)

    step, t0, history = 0, time.time(), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in loader:
            p, y, valid = run(batch)
            loss = nn.functional.huber_loss(p[valid], y[valid])
            if not torch.isfinite(loss):
                sys.exit(f"step {step + 1}: the loss is not finite, stopped")
            opt.zero_grad(set_to_none=True)
            amp.scale(loss).backward()
            amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp.step(opt)
            amp.update()
            sched.step()
            step += 1
            running.append(loss.item())
            if step % LOG_EVERY == 0 or step == steps:
                speed = step * args.batch / (time.time() - t0)
                print(f"epoch {epoch}/{args.epochs} step {step}/{steps} | loss {np.mean(running[-LOG_EVERY:]):.4f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | {speed:.1f} samples/s | ETA {(steps - step) * args.batch / speed / 60:.1f} min",
                      flush=True)
        history.append(float(np.mean(running)))
        print(f"epoch {epoch} done | mean training loss {history[-1]:.4f}", flush=True)

    model.eval()
    nb = len(BANDS)
    names = [f"{a:g}-{b:g}" for a, b in BANDS]
    loader = torch.utils.data.DataLoader(Days(args.envelopes, rest, stations, scale, have), batch_size=args.batch, num_workers=args.workers)
    rows, before, before_rows, losses = [], [], [], []
    rest = rest.reset_index(drop=True)
    with torch.no_grad():
        for batch in loader:
            p, y, valid = run(batch)
            losses.append(nn.functional.huber_loss(p[valid], y[valid]).item())
            res = torch.where(valid, y - p, torch.nan).cpu().numpy()                      # (B, 1440, 2 * bands)
            for j, i in enumerate(batch[6].tolist()):
                r, e = rest.iloc[i], res[j]
                ok = np.isfinite(e[:, 0])
                if ok.sum() < 60:
                    continue
                row = dict(date=r["date"], station=r["station"], role=r["role"], sources=r["sources"], minutes=int(ok.sum()))
                row.update({f"mean_{n}": float(np.nanmean(e[:, k])) for k, n in enumerate(names)})
                row.update({f"max95_{n}": float(np.nanpercentile(e[:, nb + k], 95)) for k, n in enumerate(names)})
                rows.append(row)
                if r["role"] == "before":
                    before.append(e.astype(np.float16))
                    before_rows.append(dict(date=r["date"], station=r["station"]))
    pd.DataFrame(rows).to_csv(out / "surprise.tmp", index=False)
    (out / "surprise.tmp").replace(out / "surprise.csv")
    if before:
        np.savez_compressed(out / "before.npz", residual=np.stack(before))
        pd.DataFrame(before_rows).to_csv(out / "before.csv", index=False)
    torch.save(dict(model=model.state_dict(), stations=stations, scale={s: [v.tolist() for v in scale[s]] for s in stations}),
               out / "model.pt")
    (out / "run.json").write_text(json.dumps(dict(
        argv=sys.argv, settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commit=commit(), versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        device=device, parameters=n_params, stations=stations, events=[str(e) for e in events],
        samples={k: int(v) for k, v in table["role"].value_counts().items()}, training_samples_used=len(train),
        training_loss_per_epoch=history, loss_not_trained_on=float(np.mean(losses)) if losses else math.nan,
        before_days=BEFORE_DAYS, after_days=AFTER_DAYS, min_sources=MIN_SOURCES, clip=CLIP), indent=1))
    (out / "DONE").touch()
    print(f"DONE | loss on the days not trained on {np.mean(losses):.4f} (last training epoch {history[-1]:.4f}) | "
          f"{len(rows)} predicted (day, station) in {out}", flush=True)


if __name__ == "__main__":
    main()
