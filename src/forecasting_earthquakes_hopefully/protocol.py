"""The question, as fixed on 2026-11-17 before any recording was read for it.

For a station and a day: will an earthquake of magnitude MIN_MAGNITUDE or above occur within RADIUS_KM of the
station in the next LEAD_DAYS days?
"""
from pathlib import Path

MIN_MAGNITUDE = 5.0     # equal or greater
RADIUS_KM = 50.0        # epicentre to station
LEAD_DAYS = 14

CLEAN_DAYS = 60         # an event is "clean" at a station when no other event of the question lies within
                        # RADIUS_KM of the station in the CLEAN_DAYS days before it

CATALOG = Path(__file__).resolve().parents[2] / "catalog"


def held_out_stations():
    """The stations whose recordings are not read until the freeze (catalog/held_out_stations.txt)."""
    return set((CATALOG / "held_out_stations.txt").read_text().split())
