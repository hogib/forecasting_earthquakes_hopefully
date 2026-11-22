import datetime as dt

import numpy as np
from scipy.signal import butter, sosfiltfilt

from forecasting_earthquakes_hopefully import correlations, velocity


def noise(n, rng, fs=50):
    return sosfiltfilt(butter(4, (0.5, 5), btype="bandpass", fs=fs, output="sos"), rng.normal(size=n))


def test_correlation_finds_the_delay_and_ignores_loudness():
    rng, fs = np.random.default_rng(0), 50
    n = correlations.SEGMENT * fs
    b = noise(n + 100, rng)
    a = np.roll(b, 2 * fs)[:n]                                   # a follows b by 2 s
    b = b[:n]
    c = correlations.correlate(correlations.prepare(a, fs), correlations.prepare(b, fs), correlations.SEGMENT * correlations.FS)
    lag = (np.argmax(c) - correlations.MAX_LAG * correlations.FS) / correlations.FS
    assert abs(lag - 2.0) <= 1 / correlations.FS
    loud = correlations.correlate(correlations.prepare(1000 * a, fs), correlations.prepare(b, fs), correlations.SEGMENT * correlations.FS)
    assert np.allclose(c, loud, atol=1e-9)


def test_day_needs_whole_pieces_on_every_component():
    rng, fs = np.random.default_rng(1), 50
    day = {c: noise(correlations.DAY * fs, rng).astype(np.float32) for c in "ZNE"}
    c, used, total = correlations.day_correlations(day, fs)
    assert c.shape == (3, 2 * correlations.MAX_LAG * correlations.FS + 1) and used == total == 48
    day["N"][::correlations.SEGMENT * fs // 2] = np.nan           # a hole in every piece of one component
    assert correlations.day_correlations(day, fs)[0] is None


def test_stretch_recovers_a_velocity_change():
    rng = np.random.default_rng(2)
    n = 2 * correlations.MAX_LAG * correlations.FS + 1
    t = (np.arange(n) - (n - 1) / 2) / correlations.FS
    fine = np.linspace(t[0] * 1.1, t[-1] * 1.1, 40 * n)
    wave = sosfiltfilt(butter(4, (0.5, 2), btype="bandpass", fs=40 * correlations.FS / 1.1, output="sos"), rng.normal(size=len(fine)))
    reference = np.interp(t, fine, wave)
    keep = (np.abs(t) >= velocity.LAGS[0]) & (np.abs(t) <= velocity.LAGS[1])
    for true in (0.004, -0.007, 0.0):
        e, cc = velocity.stretch(reference, np.interp(t * (1 + true), fine, wave), keep)
        assert abs(e - true) < 2e-4 and cc > 0.99


def test_eras_split_at_long_holes():
    day = lambda k: dt.date(2020, 1, 1) + dt.timedelta(days=k)
    assert velocity.eras([day(0), day(1), day(100), day(400), day(401)]) == [[0, 1, 2], [3, 4]]


def test_forecast_features_use_the_past_only_and_find_a_change():
    rng = np.random.default_rng(3)
    n = 2 * correlations.MAX_LAG * correlations.FS + 1
    t = (np.arange(n) - (n - 1) / 2) / correlations.FS
    fine = np.linspace(t[0] * 1.1, t[-1] * 1.1, 40 * n)
    wave = sosfiltfilt(butter(4, (0.5, 4), btype="bandpass", fs=40 * correlations.FS / 1.1, output="sos"), rng.normal(size=len(fine)))
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=k) for k in range(80)]
    speed = [0.0 if k < 70 else 0.005 for k in range(80)]          # 0.5 % faster from day 70 on
    corr = np.stack([np.stack([np.interp(t * (1 + s), fine, wave)] * 3) for s in speed])
    got = velocity.forecast_features(dates, corr)
    assert dates[32] not in got and dates[33] in got                 # 14 recent days and MIN_BASE base days
    assert abs(got[dates[69]]["dvv14_1-2"]) < 0.02 and got[dates[79]]["dvv5_1-2"] > 0.4
    later = corr.copy()
    later[75:] = rng.normal(size=later[75:].shape)                   # the future changes, the past features do not
    again = velocity.forecast_features(dates, later)
    assert again[dates[72]] == got[dates[72]]
