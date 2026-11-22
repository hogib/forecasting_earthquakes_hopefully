import datetime as dt
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from forecasting_earthquakes_hopefully import encoder, minutes, train, windows


def make(tmp_path, days=90, seed=0):
    """Two stations in each of two groups; the positive windows are louder in their last ten days."""
    rng = np.random.default_rng(seed)
    start, rows = dt.date(2021, 1, 1), []
    (tmp_path / "minutes").mkdir()
    for k, (station, group) in enumerate([("AAA", "g1"), ("BBB", "g1"), ("CCC", "g2"), ("DDD", "g2")]):
        x = (rng.normal(size=(days, minutes.CHANNELS, 1440)) * 0.1 + 2.0 + k).astype(np.float32)   # each station its own level
        event = 60 + 5 * k
        x[event - 10:event, :] += 0.5
        x[20, :, 100:400] = np.nan                                             # a gap
        dates = [start + dt.timedelta(days=i) for i in range(days)]
        np.savez(tmp_path / "minutes" / f"{station}.npz", dates=np.array([d.isoformat() for d in dates]), x=x)
        for i, d in enumerate(dates):
            label = int(event - 7 <= i < event)
            rows.append(dict(station=station, region=group, date=d.isoformat(), label=label, dropped=int(i >= event),
                             event_id=f"e{k}" if label else "", days_to_event="", group=group))
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    return tmp_path / "minutes", tmp_path / "labels.csv"


def test_summarise_keeps_a_spike_in_the_maximum():
    e = np.ones((3, 5, 86400), dtype=np.float32)
    e[0, 2, 125] = 100.0
    e[1, :, 600:660] = np.nan
    s = minutes.summarise(e)
    assert s.shape == (30, 1440)
    assert np.isclose(s[15 + 2, 2], 2.0) and s[2, 2] < 0.5 and np.isclose(s[15 + 2, 3], 0.0)
    assert np.isnan(s[5:10, 10]).all() and np.isnan(s[20:25, 10]).all() and np.isfinite(s[0:5, 10]).all()


def test_windows_are_on_their_own_scale(tmp_path):
    m, lab = make(tmp_path)
    w = windows.Windows(m, windows.read_labels(lab))
    assert w.skipped["base_too_short"] > 0 and len(w) > 0                       # the first days have no window
    assert not w.window(0)[2].all()                                             # the earliest window lacks days
    z, mask, present = w.window(len(w) - 1)
    assert z.shape == (28, 30, 1440) and mask.shape == (28, 3, 1440) and present.all()
    level = {s: np.abs(w.window(int(i))[0][:14]).mean() for s, i in w.table.groupby("station").head(1)[["station"]].assign(i=lambda t: t.index).values}
    assert max(level.values()) < 1.5                                            # the stations' own levels are gone
    loud = w.loudness().merge(w.table, on=["station", "date"])
    assert loud[loud["label"] == 1]["loud_2-5"].mean() > loud[loud["label"] == 0]["loud_2-5"].mean() + 1
    z1 = windows.Windows(m, windows.read_labels(lab), "Z")
    assert z1.window(0)[0].shape == (28, 10, 1440) and z1.window(0)[1].shape == (28, 1, 1440)


def test_models_have_the_right_shapes_and_the_hidden_part_is_not_seen():
    torch.manual_seed(0)
    enc = encoder.DayEncoder(30, 3)
    z, mask = torch.randn(4, 30, 1440), torch.ones(4, 3, 1440)
    assert enc(z, mask).shape == (4, 64)
    hidden = torch.zeros(4, encoder.SLOTS, dtype=torch.bool)
    hidden[:, 10:20] = True
    z2 = z.clone()
    z2[:, :, 150:300] = 9.0                                                     # only hidden minutes change
    assert torch.allclose(enc.tokens(z, mask, hidden), enc.tokens(z2, mask, hidden))
    rec = encoder.Reconstructor(enc, 30, np.array([k for _ in (0, 1) for k in range(3) for _ in range(5)]))
    assert rec.loss(z, mask).ndim == 0
    assert encoder.Classifier()(torch.randn(5, 28, 64), torch.ones(5, 28)).shape == (5,)


def test_the_embargo_drops_training_windows_next_to_left_out_days():
    day = lambda i: dt.date(2021, 1, 1) + dt.timedelta(days=i)
    tr = pd.DataFrame(dict(station=["A", "A", "A", "B"], date=[day(0), day(70), day(120), day(71)]))
    te = pd.DataFrame(dict(station=["A", "A"], date=[day(100), day(110)]))
    assert train.embargoed(tr, te).tolist() == [False, True, True, False]


def test_a_planted_signal_is_found_end_to_end(tmp_path):
    m, lab = make(tmp_path)
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, "-m", "forecasting_earthquakes_hopefully.train", str(m), str(lab), str(out),
                        "--pretrain-epochs", "1", "--epochs", "20", "--device", "cpu"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "DONE").exists() and "ETA" in r.stdout
    s = pd.read_csv(out / "summary.csv").iloc[0]
    assert s["auc"] > 0.8, r.stdout[-1500:]


def test_neighbours_pool_the_other_stations_of_the_day():
    table = pd.DataFrame(dict(region=["a", "a", "a", "b"], date=[1, 1, 2, 1]))
    vec = {0: torch.ones(3, 4), 1: 3 * torch.ones(3, 4), 2: torch.ones(3, 4), 3: torch.ones(3, 4)}
    pres = {r: torch.ones(3) for r in range(4)}
    pres[1] = torch.tensor([1.0, 0.0, 1.0])
    out = train.neighbours(table, [0, 1, 2, 3], vec, pres)
    assert out[0].shape == (3, 9)
    assert out[0][0].tolist() == [3.0] * 8 + [1.0] and out[0][1].tolist() == [0.0] * 9      # station 1 absent on day 1
    assert out[1][0].tolist() == [1.0] * 8 + [1.0]
    assert out[2].abs().sum() == 0 and out[3].abs().sum() == 0                              # nobody else that day
    assert encoder.Classifier(4, extra=9)(torch.randn(2, 3, 4), torch.ones(2, 3), torch.randn(2, 3, 9)).shape == (2,)


def test_by_station_counts_per_station():
    from forecasting_earthquakes_hopefully import score
    s = pd.DataFrame(dict(station=["A", "A", "A", "B", "B"], region=["r"] * 5, group=["g1", "g1", "g2", "g1", "g1"],
                          label=[1, 0, 0, 0, 0], score=[2.0, 1.0, 0.0, 5.0, 4.0]))
    s["alarm@0.1"] = [True, False, True, False, False]
    out = score.by_station(s).set_index("station")
    assert out.loc["A", ["days", "positive", "groups_positive", "auc", "recall", "alarm"]].tolist() == [3, 1, 1, 1.0, 1.0, 0.5]
    assert out.loc["B", "positive"] == 0 and np.isnan(out.loc["B", "auc"]) and out.loc["B", "alarm"] == 0.0


def test_shape_ignores_loudness():
    from forecasting_earthquakes_hopefully import day_cnn
    x = np.random.default_rng(0).normal(size=(2, 30, 8))
    louder = x + 1.7                                        # every envelope 50 times larger: log10 shifts by a constant
    assert np.allclose(day_cnn.shape(x), day_cnn.shape(louder))
    assert not np.allclose(day_cnn.shape(x)[:, :15], 0)
