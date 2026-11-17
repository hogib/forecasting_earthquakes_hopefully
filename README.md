# forecasting_earthquakes_hopefully

For a seismic station and a day: will an earthquake of magnitude 5 or above occur within 50 km of the station
in the next 14 days? The question and its numbers are fixed in `src/forecasting_earthquakes_hopefully/protocol.py`.

- `catalog/`: the station list and the earthquake list (see `catalog/PROVENANCE.md`), and the held-out stations,
  whose recordings are not read until a freeze is declared.
- `survey/`: which days of which stations are on disk and which earthquakes of the question lie near them,
  written by

      uv run python -m forecasting_earthquakes_hopefully.survey survey ROOT [ROOT ...]

  from file names only (no recording is opened).

Heavy computation runs on the server; `data/`, `runs/` and `logs/` are not tracked.
