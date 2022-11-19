# europe-historical-geojson

This project creates a desk board for strategy games
based on Europe's historical map.

I started this project to teach my kids the history
from the Congress of Vienna to the First World War.

It is incredible that the core dataset for plotting
maps based on a hundred years of wars between the Old
Continent's countries comes actually from the GISCO
dataset of the European Union maps.

## Build Empires boundaries

Pick a map using for example

```
# default is mappe.yaml
export MAPPE=mappe-1915.yaml
```

To build empire boundaries and avoid stressing OpenStreetmap
run

```
pytest -k save
```

Then build maps with

```
python -mmappe
```

this will create 3 files:

```
/tmp/full-board.png
/tmp/label-board.png
/tmp/risk-board.png
```
