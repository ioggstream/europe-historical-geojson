from io import StringIO
import geopandas as gpd
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from requests import get
from shapely.ops import cascaded_union
import yaml

def area(label):
    return get(
        f"https://gisco-services.ec.europa.eu/distribution/v2/nuts/distribution/{label}-region-10m-3857-2021.geojson"
    )


def area_frame(label):
    return gpd.read_file(area(label).content.decode())

""":return
	lightskyblue	#87CEFA	rgb(135,206,250)
 	skyblue	#87CEEB	rgb(135,206,235)
 	deepskyblue	#00BFFF	rgb(0,191,255)
 	lightsteelblue	#B0C4DE	rgb(176,196,222)
 	dodgerblue	#1E90FF	rgb(30,144,255)
 	cornflowerblue	#6495ED	rgb(100,149,237)
 	steelblue	#4682B4	rgb(70,130,180)
 	cadetblue	#5F9EA0	rgb(95,158,160)
"""

def join_areas(areas):
    area_frames = [area_frame(label) for label in areas]
    return gpd.GeoSeries(cascaded_union([x.geometry[0] for x in area_frames]))

maps = yaml.load_safe(Path("mappe.yaml").read_text())
italia = maps['Italia']

for k, v in italia.items():
  t = join_areas(v)
  t.plot(cmap='winter')

