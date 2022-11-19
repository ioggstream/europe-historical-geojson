import logging
from functools import partial
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import requests_cache
from contextily import add_basemap
from geopandas import GeoDataFrame, GeoSeries
from matplotlib import pyplot as plt
from pandas import DataFrame
from requests import get
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import cascaded_union

from .constants import *
from .utils import (
    Config,
    annotate_coords,
    baricenter,
    geoline,
    get_polygons,
    log,
    maps,
    seas,
)

FONT_REGION = "eufm10"
FONT_ARIAL = "DejaVu Sans"

FONT_REGION = "Liberation Sans"
FONT_REGION_COLOR = "black"
matplotlib.rcParams["pdf.fonttype"] = 42

log = logging.getLogger(__name__)
requests_cache.install_cache("demo_cache")


COUNTRIES = tuple(maps().keys())




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





class State:
    def __init__(self, state_name, config):
        self.name = state_name
        self.config = config
        self._state = config.maps[state_name]


    def get_regions(self) -> dict:
        """@returns: a dict containing all the regions' geometries.
        """
        regions = self._state["regions"]
        return {k: join_areas(v["codes"]) for k, v in regions.items()}

    def get_state(self, cache=True, save=False) -> GeoDataFrame:
        """:return a WGS84 geodataframe eventually intersected with the rest"""
        state_config = self._state
        translate = state_config.get("translate", [0, 0])
        scale = state_config.get("scale", [1.0, 1.0])

        f = self.name.replace("\n", " ")
        cache_file = Path(self.config.get_cache_filename(f))
        if cache and cache_file.exists():
            log.warning(f"Reading from {cache_file}")
            ret = gpd.read_file(cache_file.open())
            assert ret.crs == EPSG_4326_WGS84
            if not save:
                ret.geometry = ret.geometry.affine_transform([scale[0], 0, 0, scale[1], translate[0], translate[1]])

            return ret
        regions = list(self.get_regions().items())
        n, t = regions[0]

        df = DataFrame({"name": [n], "state": [self.name]})
        ret = gpd.GeoDataFrame(df, geometry=t, crs=EPSG_4326_WGS84)
        for n, t in regions[1:]:
            ret = ret.append(
                gpd.GeoDataFrame(
                    DataFrame({"name": [n], "state": [self.name]}),
                    geometry=t,
                    crs=EPSG_4326_WGS84,
                )
            )
        ret = ret.reset_index()

        #
        # Restrict state borders using a specific geojson.
        #
        if borders:= self.get_historical_borders():
            ret.geometry = ret.geometry.intersection(borders)

        ret = ret.set_crs(EPSG_4326_WGS84)
        if not save:
            ret.geometry = ret.geometry.affine_transform([scale[0], 0, 0, scale[1], translate[0], translate[1]])
        assert ret.crs == EPSG_4326_WGS84
        return ret

    def get_historical_borders(self):
        """Evaluate the country borders from the state config.
            It is computed as the union of all the identifiers containes int the country-borders key.
        """

        def _open_geojson_or_nuts(label_or_geojson):
            if label_or_geojson.endswith("json"):
                return gpd.read_file(open(label_or_geojson), crs=EPSG_4326_WGS84).unary_union
            return get_area(label_or_geojson).unary_union

        geo_config = self._state.get("country-borders")
        geo_config = geo_config if isinstance(geo_config, list) else [geo_config]

        borders = None
        for g in  geo_config:
            new_borders = _open_geojson_or_nuts(g)
            borders = new_borders if borders is None else borders.union(new_borders)
        return borders




#
# Render maps.
#
from multiprocessing import Manager

manager = Manager()
state_archive = manager.dict()
from .utils import intersect


def render(
    gdfm,
    facecolor1="blue",
    facecolor2="blue",
    ax=None,
    plot_labels=True,
    plot_geo=True,
    plot_cities=True,
    plot_state_labels=True,
    plot_state_labels_only=False,
    cities=None,
):
    global state_archive
    cities = cities or []
    empire = gdfm
    my_crs = MY_CRS

    state_label = gdfm.state.values[0]
    state_label_font_size = 48
    config = Config()

    # FIXME: there's a problem somewhere with the German empire.
    if state_label != "Deutschland":
        empire = intersect(empire, config.get_europe())

    state_archive[state_label] = empire
    if plot_state_labels_only:
        plot_cities = False
        plot_labels = False

    state_center = baricenter(empire.to_crs(my_crs))
    ax.annotate(
        text=state_label,
        xy=state_center,
        fontsize=state_label_font_size,
        horizontalalignment="center",
        verticalalignment="center",
        color="white",
        fontname=FONT_REGION,
        alpha=0.7 if plot_state_labels else 0,
    )

    for region_name in empire.name:
        region = empire[empire.name == region_name]
        togli_isolette(region, 0.3)
        empire[empire.name == region_name] = region

        if plot_labels:
            try:
                annotate_region(region, ax=ax)
            except:
                raise

    if plot_cities:
        for city in cities:
            annotate_location(**city, state_label=state_label, ax=ax)

    # Limit the map to EU and convert to 3857 to improve printing.
    # empire = gpd.overlay(empire, _get_europe(), how='intersection')
    empire = empire.to_crs(my_crs)

    # Draw borders with different colors.
    if plot_geo:
        empire.plot(
            ax=ax, edgecolor="black", facecolor=facecolor2, linewidth=2, alpha=1.0
        )
        empire.plot(
            ax=ax, edgecolor="black", facecolor=facecolor1, linewidth=0, alpha=0.5
        )
    else:
        empire.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0, alpha=0)

    return empire


def annotate_region(
    region,
    text=None,
    xytext=None,
    fontname=FONT_REGION,
    color=FONT_REGION_COLOR,
    ax=plt,
):
    state_label = region.state.values[0]
    region_name = region.name.values[0]
    point = baricenter(region)
    region_config = maps()[state_label]["regions"][region_name]
    region_label = region_config.get("label", region_name)
    region_label_options = region_config.get("label_options", {})
    rotation = region_label_options.get("rotation", 0)
    horizontalalignment = region_label_options.get("ha", "center")
    fontsize = 20
    padding = [region_label_options.get("x", 0), region_label_options.get("y", 0)]
    annotate_coords(
        ax=ax,
        text=text or region_label,
        xy=point,
        xytext=(i * fontsize for i in padding) if xytext is None else xytext,
        horizontalalignment=horizontalalignment,
        verticalalignment="center",
        fontsize=fontsize,
        color=color,
        fontname=fontname,
        # fontname="URW Bookman", color="black", fontsize=16,
        # fontstyle="italic",
        state_label=None,
        rotation=rotation,
        textcoords="offset points",
    )


def render_state(
    state_label, ax, plot_labels=True, plot_geo=True, plot_cities=True, **kwargs
):
    state_area = get_state(state_label)
    state_config = maps()[state_label]
    color_config = state_config["config"]
    cities = state_config.get("citta", [])
    render(
        state_area,
        ax=ax,
        plot_labels=plot_labels,
        plot_geo=plot_geo,
        plot_cities=plot_cities,
        cities=cities,
        **color_config,
        **kwargs,
    )
    return state_area



def render_seas(ax=None):
    for s in seas():
        annotate_coords(ax=ax, **s)


def test_addmap(ax):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 20)
    render_state("Ottomano", ax=ax)
    add_basemap(ax)


def render_links(ax, links_):
    # import pdb; pdb.set_trace()
    for link in links_:
        src, dst = link
        # src = find_region(src)
        # dst = find_region(dst)
        found_region = find_region(src)
        if found_region is not None:
            src = baricenter(found_region)
        else:
            src = tuple(geolocate(src))

        found_region = find_region(dst)
        if found_region is not None:
            dst = baricenter(found_region)
        else:
            dst = tuple(geolocate(dst))

        print("linea", src, dst)
        if all(x is not None for x in (src, dst)):
            line = geoline(src, dst)
            line = line.set_crs(EPSG_4326_WGS84).to_crs(MY_CRS)
            line.plot(ax=ax, color="black", linewidth=2, linestyle="dashed", zorder=1)

def get_state_archive():
    global state_archive
    if len(state_archive):
        return state_archive

    for c in COUNTRIES:
        state_ = get_state(c)
        state_archive.update({
            k: state_[state_.name==k] for k in state_.name.values
        })
    return state_archive


def render_board(countries=COUNTRIES, background=False, plot_cities=True, render_net=True, render_links=False, **kwargs):
    fig_risk, risk_board = get_board()
    fig_label, label_board = get_board()
    fig_full, full_board = get_board()
    fig_links, links_board = get_board()
    if background:
        eu = _get_europe().to_crs(MY_CRS)
        eu.plot(ax=full_board, facecolor="lightblue")
        eu.plot(ax=risk_board, facecolor="lightblue")

    countries = countries or COUNTRIES
    get_state_archive()
    if render_links:
        render_links(ax=full_board, links_=Config().links)
        render_links(ax=links_board, links_=Config().links)

    with Pool(processes=20) as pool:
        #    pool.map(partial(render_state, ax=risk_board, plot_geo=True, plot_labels=False, plot_cities=False, plot_state_labels=False), countries)
        pool.map(
            partial(
                render_state,
                ax=full_board,
                plot_geo=True,
                plot_labels=True,
                plot_cities=plot_cities,
                plot_state_labels=False,
            ),
            countries,
        )
        #    pool.map(partial(render_state, ax=label_board, plot_geo=False, plot_labels=True, plot_cities=True,  plot_state_labels=False), countries)
    #    pool.map(partial(render_state, ax=links_board, plot_geo=False, plot_labels=False, plot_cities=False, plot_state_labels=False), countries)

#    render_seas(ax=full_board)

    if render_net:
        add_neighbour_net(risk_board)

    # add_basemap(full_board, crs=str(MY_CRS), )
    suffix = get_suffix()
    cfg = dict(dpi=92, bbox_inches="tight", transparent=True)
    fig_full.savefig(f"/tmp/full-board-{suffix}.png", **cfg)
    fig_risk.savefig(f"/tmp/risk-board-{suffix}.png", **cfg)
#    fig_label.savefig(f"/tmp/label-board-{suffix}.png", **cfg)
#    fig_links.savefig(f"/tmp/links-board-{suffix}.png", **cfg)


def add_neighbour_net(ax):
    log.warning("Rendering net.")
    nbr = {}
    df = get_state(COUNTRIES[0])
    for x in COUNTRIES[1:]:
        df = df.append(get_state(x))
    df = df.intersects(_get_europe())
    prepare_neighbor_net(df, nbr)
    plot_net(nbr, ax)
    df.plot(ax=ax, color='blue')

    for region_id, values in nbr.items():
        annotate_coords(
                values["coords"],
                f"w: {values['count']}",
                textcoords="offset points",
                # xytext=(-20, -20),
                ax=ax
            )


def get_board():
    fig, board = plt.subplots(1, 1)
    plt.tight_layout(pad=0.05)
    fig.set_size_inches(cm2inch(90, 60), forward=True)
    fig.set_dpi(300)
    board.set_axis_off()
    return fig, board


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


def save_state(gdf: GeoDataFrame):
    state_label = gdf.state.values[0]

    for r in gdf.name:
        region_df = gdf[gdf.state == r]
        if region_df.empty:
            log.warning(f"Region is empty {r}")
            continue
        region_df.to_file(f"tmp-{state_label}-{r}.geojson", driver="GeoJSON")


def download_only():
    for state_label in COUNTRIES:
        get_regions(state_label)


def find_region(region_name):
    global state_archive
    for state, gdf in state_archive.items():
        region = gdf[gdf.name == region_name]
        if not region.empty:
            return region
    return None
