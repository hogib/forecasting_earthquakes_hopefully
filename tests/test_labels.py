import datetime as dt

import pandas as pd

from forecasting_earthquakes_hopefully import labels, protocol


def events(*rows):
    return pd.DataFrame([dict(id=f"e{i}", time=pd.Timestamp(t), magnitude=m, latitude=lat, longitude=38.0)
                         for i, (t, m, lat) in enumerate(rows)])


def test_only_the_first_event_of_a_sequence_counts():
    near = labels.station_events(38.0, 38.0, events(("2020-01-24 17:55", 6.7, 38.1), ("2020-01-25 16:30", 5.1, 38.1),
                                                    ("2020-06-05 18:06", 5.1, 38.2), ("2020-03-01", 4.9, 38.0),
                                                    ("2020-03-02", 6.0, 39.0)))
    assert [(e[1], e[4]) for e in near] == [("e0", True), ("e1", False), ("e2", True)]


def test_label_and_dropped_days():
    near = labels.station_events(38.0, 38.0, events(("2020-01-24 17:55", 6.7, 38.1), ("2020-01-25 16:30", 5.1, 38.1)))
    day = lambda s: labels.label_day(dt.date.fromisoformat(s), near)
    assert day("2020-01-09") == (0, False, "", "")                       # 15 days before
    assert day("2020-01-10") == (1, False, "e0", protocol.LEAD_DAYS)
    assert day("2020-01-23") == (1, False, "e0", 1)
    assert day("2020-01-24")[1] and day("2020-03-25")[1]                 # the event day, 60 days after the second
    assert day("2020-03-26") == (0, False, "", "")


def test_longer_drop_keeps_out_the_recovery():
    near = labels.station_events(38.0, 38.0, events(("2020-01-24 17:55", 6.7, 38.1), ("2020-06-05 18:06", 5.1, 38.2)))
    day = lambda s, n: labels.label_day(dt.date.fromisoformat(s), near, drop_days=n)
    assert day("2020-05-25", protocol.CLEAN_DAYS) == (1, False, "e1", 11)     # 4 months after: a positive day
    assert day("2020-05-25", 180)[1]                                          # inside 180 days: dropped
    assert day("2020-12-02", 180)[1] and day("2020-12-03", 180) == (0, False, "", "")      # 180 and 181 days after e1
