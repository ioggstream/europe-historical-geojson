import json
import logging
import time
from io import StringIO
from itertools import product
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import requests_cache
import yaml
from contextily import add_basemap
from geopandas import GeoDataFrame, GeoSeries
from matplotlib import pyplot as plt
from pandas import DataFrame
from requests import get
from requests_cache import CachedSession
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import cascaded_union

log = logging.getLogger(__name__)
requests_cache.install_cache("demo_cache")

config = lambda: yaml.safe_load(Path("mappe.yaml").read_text())
maps = lambda: config()["maps"]

COUNTRIES = (
    "Italia",
    "France",
    "Asburgici",
    "Ottomano",
    "Regno Unito",
    "Indipendenti",
    "Prussia",
    "Russia",
)

MY_EPSG = 3003
# MY_EPSG = "EPSG:4326"
MY_EPSG = 2196
MY_EPSG = 3395

# Proiezione di Peters con aree riadattate.
MY_CRS = pyproj.Proj(
    proj="cea", lon_0=0, lat_ts=45, x_0=0, y_0=0, ellps="WGS84", units="m"
).srs
# MY_CRS = f'EPSG:{MY_EPSG}'


def annotate_city(address):
    coords = get_city_location(address)
    map_coords = point_coords(*coords)
    plt.annotate(text="◉", xy=map_coords, fontsize=24)  # address.split(',', 1)[0],


def get_city_location(address):
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="github.com/ioggstream")
    ret = geolocator.geocode(address)
    return ret.point.longitude, ret.point.latitude


from shapely.geometry import Point


def point_coords(x=24, y=41):
    x = Point(x, y)
    points = gpd.GeoDataFrame({"geometry": [x]}, crs="EPSG:4326").to_crs(MY_CRS)
    return points.geometry[0].coords[:][0]


def _get_europe():
    europe = config()["europe_borders"]
    eu_area = gpd.read_file(StringIO(json.dumps(europe)))
    eu_area = eu_area.set_crs("EPSG:4326")
    return eu_area


def filter_noise(shape: MultiPolygon):
    return MultiPolygon([p for p in shape if p.area > 1_010_202_521])


def togli_isolette(area, base=1):
    for i, poli in enumerate(list(area.geometry)):
        if isinstance(poli, MultiPolygon):
            area.geometry[i] = MultiPolygon([k for k in poli if k.area > base])
    return area


def togli_isolette_2(area, base):
    print(id(area))
    entity = getattr(area, "geometry", area)
    for i, poli in enumerate(entity):
        if isinstance(poli, MultiPolygon):
            print(i, max(k.area for k in poli))
            entity[i] = MultiPolygon([k for k in poli if k.area > base])
        if isinstance(poli, Polygon):
            print("poli", i, poli.area)


def get_polygons(label, retry=0):
    """
    Si collega ad internet e scarica i poligoni associati alla label.
    """
    coord_type = 4326
    osm_fr = "http://polygons.openstreetmap.fr"
    osm_tw = "https://api06.dev.openstreetmap.org"
    base = "https://gisco-services.ec.europa.eu/distribution/v2"
    log.warning("Retrieving %r", label)

    if str(label).isdigit():
        u = f"{osm_fr}/get_geojson.py?id={label}&params=0"
        # u=f"https://nominatim.openstreetmap.org/details.php?osmtype=R&osmid={label}&class=boundary&format=json"
        if retry:
            # If the entry is not in the geojson remote cache, trigger a regeneration
            #  pinging poligons.
            requests_cache.core.get_cache().delete_url(u)
            get(f"http://polygons.openstreetmap.fr/?id={label}")
        res = get(u, proxies={"http": "socks5://localhost:11111"})
        log.warning("Request from cache: %r %r", u, res.from_cache)
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


def get_axis():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    return fig, ax


def get_area(label) -> GeoSeries:
    for i in range(3):
        polygons = get_polygons(label, i)
        log.warning(
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
    return gpd.read_file(polygons)


def join_areas(areas: List) -> GeoSeries:
    tolerance = 0.04
    get_areas = [get_area(label) for label in areas if label]
    ret = GeoSeries(
        cascaded_union([x.geometry[0] for x in get_areas if x is not None])
    ).simplify(tolerance=tolerance)
    return ret


def get_state(state_label) -> dict:
    territori = maps()[state_label]["territori"]
    return {k: join_areas(v) for k, v in territori.items()}


def get_state_df(state_label) -> GeoDataFrame:
    # get_state returns EPSG4236
    territori = list(get_state(state_label).items())
    n, s = territori[0]

    df = DataFrame({"name": [n]})
    ret = gpd.GeoDataFrame(df, geometry=s)
    for n, s in territori[1:]:
        ret = ret.append(gpd.GeoDataFrame(DataFrame({"name": [n]}), geometry=s))

    state_config = maps()[state_label]
    geo_config = state_config.get("country-borders")
    if geo_config:
        borders = gpd.read_file(open(geo_config)).to_crs(epsg=4326)
        ret.geometry = ret.geometry.intersection(borders)

    ret = ret.set_crs("EPSG:4326")
    return ret


def render(
    gdfm,
    facecolor1="blue",
    facecolor2="blue",
    ax=None,
    plot_labels=True,
    plot_geo=True,
    cities=None,
):
    cities = cities or []
    empire = gdfm
    epsg_projection = 3857
    my_crs = MY_CRS

    # print(empire)
    for region_name in empire.name:
        region = empire[empire.name == region_name]
        # print(region, region_name)

        togli_isolette(region, 0.2)
        empire[empire.name == region_name] = region

        if plot_labels:
            try:
                points = (
                    region.intersection(_get_europe())
                    .to_crs(my_crs)
                    .representative_point()
                )
                for p in points:
                    plt.annotate(
                        text=region_name,
                        xy=p.coords[:][0],
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=14,
                        # fontstyle="italic",
                        color="white",
                        fontname="eufm10",
                        #                    fontname="URW Bookman",
                    )
            except:
                pass

    for city in cities:
        annotate_city(city)

    # Limit the map to EU and convert to 3857 to improve printing.
    empire = gdfm.intersection(_get_europe())
    empire = empire.to_crs(my_crs)

    # Draw borders with different colors.
    if plot_geo:
        empire.plot(
            ax=ax, edgecolor="black", facecolor=facecolor2, linewidth=5, alpha=0.7
        )
        empire.plot(
            ax=ax, edgecolor="black", facecolor=facecolor1, linewidth=0, alpha=0.5
        )
    else:
        empire.plot(ax=ax, edgecolor="none", facecolor="none", linewidth=0)


def addmap(ax):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 20)
    render_state("Ottomano", ax=ax)
    add_basemap(ax)


def render_state(state_label, ax, plot_labels=True, plot_geo=True):
    state_area = get_state_df(state_label)
    state_config = maps()[state_label]
    color_config = state_config["config"]
    cities = state_config.get("citta", [])
    render(
        state_area,
        ax=ax,
        **color_config,
        plot_labels=plot_labels,
        plot_geo=plot_geo,
        cities=cities,
    )
    return state_area


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def get_board():
    fig, risk_board = plt.subplots(1, 1)
    plt.tight_layout(pad=1)
    fig.set_size_inches(cm2inch(29 * 2, 21 * 2), forward=True)
    fig.set_dpi(300)
    return fig, risk_board


def render_board(countries=COUNTRIES, background=False):
    fig, risk_board = get_board()
    fig_label, label_board = get_board()
    fig_full, full_board = get_board()
    if background:
        eu = _get_europe().to_crs(MY_CRS)
        eu.plot(ax=risk_board, facecolor="lightblue")
    from multiprocessing.pool import ThreadPool as Pool
    from functools import partial

    countries = countries or COUNTRIES
    with Pool(processes=20) as pool:
        pool.map(partial(render_state, ax=risk_board, plot_labels=False), countries)
        pool.map(
            partial(render_state, ax=label_board, plot_geo=False, plot_labels=True),
            countries,
        )
        pool.map(partial(render_state, ax=full_board), countries)
    pool.join()
    # add_basemap(full_board, crs=f"EPSG:{MY_EPSG}")
    fig.savefig("/tmp/risk-board.png", dpi=300, transparent=True)
    fig_label.savefig("/tmp/label-board.png", dpi=300, transparent=True)
    fig_full.savefig("/tmp/full-board.png", dpi=300, transparent=True)


def test_unite_maps():
    prussia = gpd.read_file(open("data/geojson/germany-1914.geojson"))
    de = get_area("DE")
    prussia.append(de)
    p3 = GeoDataFrame(geometry=[prussia.geometry.unary_union], crs=prussia.crs)
    impger = get_state_df("Prussia")
    impger.intersection(p3).plot()
    p3.to_file("germany-borders-1914.geojson", driver="GeoJSON")


def borderize(df, x):
    df_x = df[df.name == x]
    brd = df_x.geometry.exterior.unary_union
    poly = Polygon(shape(brd))
    GeoSeries(poly).to_file(f"tmp-{x}.geojson", driver="GeoJSON")


def dump_cache():
    simple_cache = requests_cache.core.get_cache()
    osm_requests = [
        (k, v[0].url)
        for k, v in simple_cache.responses.items()
        if "polygons" in v[0].url
    ]

    def _get_osm_id(url):
        return parse_qs(urlparse(url).query)["id"][0]

    for k, u in osm_requests:
        data = simple_cache.get_response_and_time(k)[0].content
        if data.startswith(b"{"):
            ("data/geojson" / Path(f"osm-{_get_osm_id(u)}.geojson")).write_bytes(data)


def download_only():
    for state_label in COUNTRIES:
        get_state(state_label)


if __name__ == "__main__":
    from sys import argv

    countries = argv[1:] if len(argv) > 1 else None
    render_board(countries)
