"""Which days are held out. A held-out day is never opened: no envelope is built from it and no label of it is
scored until the freeze is declared.

Two rules, and a day is held out when either says so:

1. The days already held out in the earlier project (catalog/held_out_days_previous.csv: the test blocks of
   cnn_lstm_reg and the days within 3 days of them), for every station of the same region. They cover the
   region's PREVIOUS period.
2. Outside those periods: the calendar is cut into blocks of BLOCK_DAYS days from ANCHOR, the same blocks at
   every station of a region, and TEST_SHARE of the blocks are held out, chosen by a hash of (SEED, region,
   block), so the choice does not depend on what is on disk or on the order of anything.

A model window may span a held-out day; the day is then missing in the input (mask 0), like a gap.
"""
import csv
import datetime as dt
import hashlib
from functools import lru_cache

from . import protocol

ANCHOR = dt.date(2000, 1, 1)
BLOCK_DAYS = 14
TEST_SHARE = 0.2
SEED = 0

REGIONS = {
    "elazig": ["ELZG", "FRT", "SVRC", "KOVA", "MAYA", "NARI", "CNG"],
    "balikesir": ["PURS", "STEP", "DURS", "BKES", "DEMI", "AKHS"],
    "van": ["TVAN", "VMUR", "GEVA", "DYDN"],
}
PREVIOUS = {   # region: first and last day that rule 1 decides alone
    "elazig": (dt.date(2015, 1, 24), dt.date(2020, 1, 24)),
    "balikesir": (dt.date(2025, 2, 10), dt.date(2025, 8, 10)),
}


def region_of(station):
    for region, stations in REGIONS.items():
        if station in stations:
            return region
    raise KeyError(f"{station} has no region in split.REGIONS")


@lru_cache
def _previous():
    out = {}
    with open(protocol.CATALOG / "held_out_days_previous.csv") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["region"], set()).add(dt.date.fromisoformat(r["date"]))
    return out


def block_held_out(region, day):
    """Rule 2 for one day."""
    block = (day - ANCHOR).days // BLOCK_DAYS
    h = hashlib.sha256(f"{SEED}:{region}:{block}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64 < TEST_SHARE


def held_out(station, day):
    if station in protocol.held_out_stations():
        return True
    region = region_of(station)
    if region in PREVIOUS and PREVIOUS[region][0] <= day <= PREVIOUS[region][1]:
        return day in _previous().get(region, set())
    return block_held_out(region, day)
