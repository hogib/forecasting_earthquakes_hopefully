# Where the files of this folder come from

`stations.csv` and `events.csv` are copies of `src/tdvms/data/stations.csv` and `src/tdvms/data/events.csv`
of https://github.com/hogib/tdvms-pull at commit 742c9a4f2fd0fcff699e7eb38ac3c5bfb5cd0a45, copied on
2026-11-17 and not changed.

- `stations.csv`: code, network, latitude, longitude, devices of the stations of the AFAD TDVMS portal.
  sha256 366d0d601bf326e6c2ab18fb9f1ec0f07f9c749ce25b690ae65cf8c7246c2bfb
- `events.csv`: the earthquakes of magnitude 5 and above in and around Turkey from 1990 on (ids of the USGS
  catalogue), time in UTC. sha256 90cfac12cb528ee38007f8e012f2420895b3ea95786c46d10ac7c87be3346dfd

`held_out_stations.txt` is written by hand: the stations of the Marmara / Silivri set. Nothing in this repo
reads their recordings until the freeze is declared.
