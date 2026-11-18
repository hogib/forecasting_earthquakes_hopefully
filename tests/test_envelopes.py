import datetime as dt

import numpy as np

from forecasting_earthquakes_hopefully import envelopes, split


def test_a_sine_lands_in_its_band():
    fs, seconds = 100.0, 600
    t = np.arange(int(fs * seconds)) / fs
    env, gaps = envelopes.band_rms(4.0 * np.sin(2 * np.pi * 3.0 * t), fs)
    assert env.shape == (len(envelopes.BANDS), seconds) and gaps == 0
    middle = env[:, 60:-60]
    assert np.allclose(middle[2], 4.0 / np.sqrt(2), rtol=0.02)          # 2-5 Hz
    assert (middle[[0, 1, 3, 4]] < 0.05).all()


def test_a_long_gap_stays_missing_and_a_short_one_is_filled():
    fs = 100.0
    x = np.random.default_rng(0).normal(size=int(fs * 900))
    x[30000:30050] = np.nan            # 0.5 s
    x[60000:61000] = np.nan            # 10 s
    env, gaps = envelopes.band_rms(x, fs)
    assert np.isfinite(env[:, 300]).all()
    assert np.isnan(env[:, 600:610]).all() and np.isfinite(env[:, 598]).all() and np.isfinite(env[:, 611]).all()
    assert gaps == 1


def test_bands_above_the_sampling_rate_are_left_out():
    env, _ = envelopes.band_rms(np.random.default_rng(1).normal(size=20 * 300), 20.0)
    assert np.isfinite(env[:3]).all() and np.isnan(env[3:]).all()


def test_span_of_a_file_name():
    assert envelopes.span("TU_ELZG_14082019_000000_04092019_000000_HH.mseed") == ("ELZG", dt.date(2019, 8, 14), dt.date(2019, 9, 4))


def test_split_is_fixed_and_holds_out_a_fifth():
    days = [dt.date(2020, 2, 1) + dt.timedelta(days=i) for i in range(1400)]
    held = [split.held_out("ELZG", d) for d in days]
    assert held == [split.held_out("FRT", d) for d in days]              # the same blocks in a region
    assert 0.12 < np.mean(held) < 0.28
    assert held != [split.held_out("TVAN", d) for d in days]


def test_previous_held_out_days_stay_held_out():
    assert split.held_out("ELZG", dt.date(2019, 12, 2)) and split.held_out("PURS", dt.date(2025, 7, 16))
    assert not split.held_out("ELZG", dt.date(2020, 1, 20))
    assert split.held_out("MARM", dt.date(2020, 1, 20))


def test_components_are_chosen_by_name_and_a_missing_one_is_none():
    import io

    from obspy import Stream, Trace, UTCDateTime
    start, fs = UTCDateTime(2020, 2, 1), 20.0
    st = Stream([Trace(np.full(int(fs * 3600), v, dtype=np.int32), dict(channel=ch, sampling_rate=fs, starttime=start + shift))
                 for ch, v, shift in (("HHN", 2, 0), ("HHZ", 1, 60))])              # N comes first in the file
    raw = io.BytesIO()
    st.write(raw, format="MSEED")
    out, rate, found = envelopes.read_components(raw.getvalue(), dt.date(2020, 2, 1), dt.date(2020, 2, 2), "ZNE")
    assert rate == fs and found == ["HHN", "HHZ"] and out["E"] is None
    z, n = out["Z"][0], out["N"][0]
    assert out["Z"][1] == "HHZ" and len(z) == int(fs * 86400)
    assert np.isnan(z[:int(fs * 60)]).all() and (z[int(fs * 60):int(fs * 3660)] == 1).all() and np.isnan(z[int(fs * 3660):]).all()
    assert (n[:int(fs * 3600)] == 2).all()


def test_filtering_in_blocks_gives_what_one_piece_gives():
    fs = 50.0
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(size=int(fs * 7200))) + 50 * rng.normal(size=int(fs * 7200))     # drifting, all bands
    x[int(fs * 2990):int(fs * 3020)] = np.nan                                                # a gap across a block edge
    x[int(fs * 5000):int(fs * 5000) + 20] = np.nan                                           # a short one, filled
    whole, gaps = envelopes.band_rms(x.astype(np.float32), fs, chunk=None)
    parts, gaps2 = envelopes.band_rms(x.astype(np.float32), fs, chunk=1000, pad=600)
    assert gaps == gaps2 == 1
    assert (np.isfinite(whole) == np.isfinite(parts)).all()
    ok = np.isfinite(whole)
    d = np.abs(np.log10(parts[ok] / whole[ok]))        # the largest differences sit at the edge of the gap, lowest band
    assert d.max() < 0.01 and np.percentile(d, 99.9) < 0.001
