import numpy as np
import pandas as pd

from forecasting_earthquakes_hopefully import rivals, score


def table(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame(dict(station=rng.choice(["A", "B", "C"], n), group=rng.choice(["g1", "g2", "g3"], n),
                          label=(rng.random(n) < 0.1).astype(int), dropped=(rng.random(n) < 0.1).astype(int)))
    t["region"], t["event_id"] = t["group"], np.where(t["label"] == 1, "e" + t["group"], "")
    t["x"] = t["label"] * 2.0 + rng.normal(size=n)
    t["noise"] = rng.normal(size=n)
    return t


def test_auc():
    assert score.auc([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
    assert score.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert np.isnan(score.auc([1, 2], [0, 0]))


def test_a_feature_that_knows_is_found_and_noise_is_not():
    t = table()
    good = score.summary(score.out_of_fold(t, rivals.logistic(["x"])))
    bad = score.summary(score.out_of_fold(t, rivals.logistic(["noise"])))
    assert good["auc"] > 0.85 and good["recall@0.1"] > 3 * good["alarm@0.1"]
    assert 0.35 < bad["auc"] < 0.65
    assert good["days"] == int((t["dropped"] == 0).sum())


def test_the_alarm_level_comes_from_the_training_days_only():
    t = table()
    t.loc[t["group"] == "g1", "x"] += 100                      # the left-out group is far louder than the others
    scored = score.out_of_fold(t, lambda train, test: test["x"].to_numpy())
    assert scored[scored["group"] == "g1"]["alarm@0.05"].all()
