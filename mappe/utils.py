import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests_cache
from geopandas import GeoDataFrame, GeoSeries
from matplotlib import pyplot as plt
from requests import get
from shapely.geometry import MultiPolygon, Point, shape
from shapely.ops import cascaded_union

from .constants import EPSG_4326_WGS84, MY_CRS

log = logging.getLogger(__name__)
ZORDER_TEXT = 1000
LARGE_CITY = "\u2299"  # "◉"


def get_axis():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    return fig, ax

def intersect(df1: GeoDataFrame, df2: GeoDataFrame) -> GeoDataFrame:
    assert df1.crs == df2.crs
    return gpd.overlay(df1, df2, how="intersection")

class Map:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.gdf = None

def prepare_neighbor_net(gdf: GeoDataFrame, nbr: dict):
    for index, row in gdf.iterrows():
        name = row[1]
        index = name
        region = gdf[gdf.name == index]
        print(index)
        nbr[index] = {"coords": baricenter(region)}
        disjoints = gdf.geometry.disjoint(row.geometry)
        print("disjoint", row, disjoints)
        neighbors = gdf[~disjoints].name.tolist()
        # neighbors = gdf[disjoints].name.tolist()
        if index in neighbors:
            neighbors.remove(index)
        print(index, neighbors)
        nbr[index]["nbr"] = neighbors
        nbr[index]["count"] = len(neighbors)


def plot_net(nbr, ax):
    for k, v in nbr.items():
        x = v["coords"]
        for d in v["nbr"]:
            y = nbr[d]["coords"]
            geoline(x, y).plot(ax=ax, color="red")


def get_area(label) -> GeoSeries:
    for i in range(3):
        polygons = get_polygons(label, i)
        log.info(
            "polygons: %r  len: %r head: %r",
            label,
            len(polygons),
            polygons[:10] if polygons else "Vuoto",
        )

        if polygons.startswith("{"):
            break
    else:
        return None
        # raise ValueError(f"Can't find polygons for {label}")
    # scale borders to simplify intersections.
    ret = gpd.read_file(polygons)
    if label == "FI":
        ret = ret.scale(xfact=1.1, yfact=1.1)
    if label == "UKMx":
        # import pdb; pdb.set_trace()
        ret = ret.scale(xfact=1, yfact=1)
    return ret


def join_areas(areas: List[str]) -> GeoSeries:
    tolerance = None
    for tolerance in (0.035,):
        try:
            get_areas = [get_area(label) for label in areas if label]
            ret = GeoSeries(
                cascaded_union([x.geometry[0] for x in get_areas if x is not None])
            ).simplify(tolerance=tolerance)
            # .scale(xfact=0.9, yfact=0.9)
            return ret
        except:
            raise
            # pass
    raise ValueError("Tolerance too high to", tolerance)


def filter_noise(shape: MultiPolygon):
    return MultiPolygon([p for p in shape if p.area > 1_010_202_521])


def togli_isolette(area, base=1):
    for i, poli in enumerate(list(area.geometry)):
        if isinstance(poli, MultiPolygon):
            area.geometry[i] = MultiPolygon([k for k in poli if k.area > base])
    return area


def get_polygons(label, retry=0):
    """
    Si collega ad internet e scarica i poligoni associati alla label.
    """

    coord_type = 4326
    osm_fr = "http://polygons.openstreetmap.fr"
    osm_tw = "https://api06.dev.openstreetmap.org"
    base = "https://gisco-services.ec.europa.eu/distribution/v2"
    log.info("Retrieving %r", label)

    if str(label).isdigit():
        u = f"{osm_fr}/get_geojson.py?id={label}&params=0"
        # u=f"https://nominatim.openstreetmap.org/details.php?osmtype=R&osmid={label}&class=boundary&format=json"
        if retry:
            # If the entry is not in the geojson remote cache, trigger a regeneration
            #  pinging poligons.
            requests_cache.core.get_cache().delete_url(u)
            get(f"http://polygons.openstreetmap.fr/?id={label}")
        res = get(u)  # , proxies={"http": "socks5://localhost:11111"})
        log.info("Request from cache: %r %r", u, res.from_cache)
        if res.content:
            return res.content.decode()
        return "{}"
    if label.startswith("http"):
        return get(label).content.decode()

    if label.startswith("file://"):
        return Path(label[7:]).read_text()

    for db, year in (("nuts", 2021), ("countries", 2020)):
        ret = get(
            f"{base}/{db}/distribution/{label}-region-10m-{coord_type}-{year}.geojson"
        )
        if ret.status_code == 200:
            break
        print(f"cannot find {ret.url}")
    return ret.content.decode()


def annotate_coords(xy, text, translate=(0,0), scale=(1,1), ax=plt, padding=(0, 0), **kwargs):
    tx, ty = translate
    a11, a22 = scale
    xy = GeoSeries(Point(xy[0], xy[1])).affine_transform([a11, 0, 0, a22, tx, ty]).geometry[0].coords[:][0]

    xytext = kwargs.get("xytext", None)
    fontsize = kwargs.get("fontsize", 24)
    if not xytext:
        kwargs["xytext"] = (i * fontsize for i in padding)
        kwargs["textcoords"] = "offset points"

    coords = np.array(xy)
    # coords += np.array(translate)
    map_coords = point_to_map_coordinates(*coords)
    # Annotate the given point with centered alignment.
    log.debug(f"annotating {text} @{xy}, {translate}, {scale}")
    ax.annotate(text=text, xy=map_coords, ha="center", va="center", zorder=ZORDER_TEXT, **kwargs)


def geolocate(address):
    from geopy.geocoders import MapQuest

    requests_cache.install_cache("geopy_cache")
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


def point_to_map_coordinates(x=24, y=41, crs=EPSG_4326_WGS84) -> List:
    """
    Convert a point in the given CRS to the map CRS.
    """
    x = Point(x, y)
    points = gpd.GeoDataFrame({"geometry": [x]}, crs=crs).set_crs(crs).to_crs(MY_CRS)
    return [x for x in points.geometry[0].coords[:][0]]


def geolocate_as_dataframe(address) -> List[GeoDataFrame]:
    coords = geolocate(address)
    return point_to_map_coordinates(*coords)


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
