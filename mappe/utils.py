import logging
import os
from functools import lru_cache
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

config = lambda: yaml.safe_load(
    Path(os.environ.get("MAPPE_YAML", "mappe.yaml")).read_text()
)
maps = lambda: config()["maps"]
seas = lambda: config()["seas"]
links = lambda: config()["links"]


import os


def get_cache_filename(state):
    suffix = get_suffix()
    return f"tmp-{suffix}-{state}.geojson"


def get_suffix():
    return os.path.basename(os.environ.get("MAPPE_YAML", "mappe.yaml"))


def get_config(fpath="mappe.yaml"):
    config = yaml.safe_load(Path(fpath).read_text())
    config.maps = config["maps"]
    config.maps = config["seas"]
    config.maps = config["links"]
    return config


def annotate_location(
    address,
    text=LARGE_CITY,
    state_label=None,
    fontsize=24,
    fontname="DejaVu Serif",
    ax=plt,
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
        ax=ax,
        **kwargs,
    )


def annotate_coords(xy, text, state_label=None, ax=plt, padding=(0, 0), **kwargs):
    log.warning(f"annotate {text} @{xy}")
    translate = (0, 0)
    if state_label:
        state_config = maps()[state_label]
        translate = state_config.get("translate", [0, 0])

    xytext = kwargs.get("xytext", None)
    fontsize = kwargs.get("fontsize", 24)
    if not xytext:
        kwargs["xytext"] = (i * fontsize for i in padding)
        kwargs["textcoords"] = "offset points"

    coords = np.array(xy)
    coords += np.array(translate)
    map_coords = point_coords(*coords)
    # Annotate the given point with centered alignment.
    log.warning(f"annotate {text} @{map_coords}, {kwargs}")
    ax.annotate(text=text, xy=map_coords, ha="center", va="center", **kwargs)


def geolocate(address):
    from geopy.geocoders import MapQuest

    log.warning(f"geolocate {address}")
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
