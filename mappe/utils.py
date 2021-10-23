import logging
import os
from functools import lru_cache
from operator import add
from pathlib import Path
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, shape

from .constants import EPSG_4326_WGS84, MY_CRS

log = logging.getLogger(__name__)

LARGE_CITY = "\u2299"  # "◉"

config = lambda: yaml.safe_load(Path("mappe.yaml").read_text())
maps = lambda: config()["maps"]
seas = lambda: config()["seas"]
links = lambda: config()["links"]


def annotate_location(
    address,
    text=LARGE_CITY,
    state_label=None,
    fontsize=24,
    fontname="DejaVu Serif",
    **kwargs,
):

    coords = geolocate(address)
    if not coords:
        log.error(f"Cannot find location: {address}, skipping")
        return None

    annotate_coords(
        coords,
        text=text,
        state_label=state_label,
        fontsize=fontsize,
        fontname=fontname,
        **kwargs,
    )


def annotate_coords(xy, text, state_label=None, **kwargs):
    translate = (0, 0)
    # adjust = [-1863.686871749116, -252.13592858798802]
    adjust = [0, 0]
    if state_label:
        state_config = maps()[state_label]
        translate = state_config.get("translate", [0, 0])
    coords = np.array(xy)
    coords += np.array(translate)
    map_coords = point_coords(*coords)
    map_coords = tuple(map(add, map_coords, adjust))
    # Annotate the given point with centered alignment.
    plt.annotate(text=text, xy=map_coords, ha="center", va="center", **kwargs)


def geolocate(address):
    from geopy.geocoders import MapQuest

    try:
        geolocator = MapQuest(
            user_agent="europe-geojson", api_key=os.environ.get("MAPQUEST_API_KEY")
        )
        ret = geolocator.geocode(address)
        return ret.point.longitude, ret.point.latitude
    except Exception as e:
        log.exception("Error reading %r", e)
        return None


def test_city_location_sea():
    geolocate("Mare adriatico")
    raise NotImplementedError


def point_coords(x=24, y=41, crs=EPSG_4326_WGS84):
    x = Point(x, y)
    points = gpd.GeoDataFrame({"geometry": [x]}, crs=crs).set_crs(crs).to_crs(MY_CRS)
    return [x for x in points.geometry[0].coords[:][0]]


def geolocate_as_dataframe(address) -> List[GeoDataFrame]:
    coords = geolocate(address)
    return point_coords(*coords)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


@lru_cache(maxsize=100)
def geoline(x, y):
    """Draw a line between two points"""
    l = {"type": "LineString", "coordinates": [list(x), list(y)]}
    return GeoSeries(shape(l))


def baricenter(s: GeoDataFrame):
    """Ritorna il  baricentro di una geometria """
    return s.unary_union.representative_point().coords[:][0]


def test_baricenter():
    gdf = gpd.read_file(Path("tmp-italia.geojson"))
    assert baricenter(gdf)
    for i in gdf.index:
        print(baricenter(gdf[gdf.index == i]))


def collega(*regions):
    """Collega due regions"""
    line = [baricenter(x) for x in regions]
    return geoline(line[0], line[1])
