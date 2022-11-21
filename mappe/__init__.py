import json
import logging
from functools import partial
from io import StringIO
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import requests_cache
import yaml
from geopandas import GeoDataFrame, GeoSeries
from matplotlib import pyplot as plt
from pandas import DataFrame
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import cascaded_union

from .constants import *
from .utils import *

FONT_REGION = "eufm10"
FONT_ARIAL = "DejaVu Sans"

FONT_REGION = "Liberation Sans"
FONT_REGION_COLOR = "black"
ZORDER_BG = 0
ZORDER_REGION_FC1 = 200
ZORDER_REGION_FC2 = 100
ZORDER_REGION_FC0 = 100
ZORDER_LINKS = 50
matplotlib.rcParams["pdf.fonttype"] = 42

log = logging.getLogger(__name__)
requests_cache.install_cache("demo_cache")

seas = NotImplementedError
render_state = NotImplementedError

class Config:
    def __init__(self, filename: str = "mappe.yaml"):
        self.config = lambda : yaml.safe_load(Path(filename).read_text())
        self.filename = filename
        self.suffix = Path(filename).stem

    @property
    def maps(self):
        return self.config()["maps"]
    @property
    def seas(self):
        return self.config()["seas"]
    @property
    def links(self):
        return self.config()["links"]
    def cache_filename(self, state: str):
        return f"tmp-{self.suffix}-{state}.geojson"
    @property
    def states(self):
        return self.maps.keys()
    def get_europe(self) -> GeoDataFrame:
        europe = self.config()["europe_borders"]
        eu_area = gpd.read_file(StringIO(json.dumps(europe)))
        eu_area = eu_area.set_crs(EPSG_4326_WGS84)
        return eu_area


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



config = Config()
COUNTRIES = tuple(config.maps.keys())


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


class State:
    def __init__(self, state_name: str, config: Config):
        self.name = state_name
        self.config = config
        self._state = config.maps[state_name]
        self._gdf = None

    def get_regions(self) -> dict:
        """@returns: a dict containing all the regions' geometries.
        """
        regions = self._state["regions"]
        return {k: join_areas(v["codes"]) for k, v in regions.items()}

    @property
    def gdf(self):
        if self._gdf is None:
            self._gdf = self.get_state()
            # FIXME: there's a problem somewhere with the German empire.
            if self.name != "Deutschland":
                self._gdf = intersect(self._gdf, self.config.get_europe())

        return self._gdf

    def save(self):
        df = self.get_state(cache=False, save=True)
        soglia_isolette = 0.2
        try:
            togli_isolette(df, soglia_isolette)
        except:
            pass
        fpath = self.config.cache_filename(self.name)
        df.to_file(fpath, driver="GeoJSON")

    def get_state(self, cache=True, save=False) -> GeoDataFrame:
        """:return a WGS84 geodataframe eventually intersected with the rest"""
        state_config = self._state
        translate = state_config.get("translate", [0, 0])
        scale = state_config.get("scale", [1.0, 1.0])

        f = self.name.replace("\n", " ")
        cache_file = Path(self.config.cache_filename(f))
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

        geo_config = self._state.get("country-borders", [])
        geo_config = geo_config if isinstance(geo_config, list) else [geo_config]

        borders = None
        for g in  geo_config:
            new_borders = _open_geojson_or_nuts(g)
            borders = new_borders if borders is None else borders.union(new_borders)
        return borders

    @property
    def cities(self):
        return self._state.get("citta", [])

    def annotate_region(self,
        region,
        text=None,
        xytext=None,
        fontname=FONT_REGION,
        color=FONT_REGION_COLOR,
        ax=plt,
    ):
        region_name = region.name.values[0]
        point = baricenter(region)
        region_config = self._state["regions"][region_name]
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
            rotation=rotation,
            textcoords="offset points",
        )

    def state_center(self, my_crs=MY_CRS):
        return baricenter(self.gdf.to_crs(my_crs))

    def render(self,
        ax, plot_labels=True, plot_geo=True, plot_cities=True, **kwargs
    ):
        color_config = self._state["config"]
        self._render(
            ax=ax,
            plot_labels=plot_labels,
            plot_geo=plot_geo,
            plot_cities=plot_cities,
            **color_config,
            **kwargs,
        )
        return self.gdf

    def _render(
        self,
        ax=None,
        facecolor1="blue",
        facecolor2="blue",
        plot_labels=True,
        plot_geo=True,
        plot_cities=True,
        plot_state_labels=True,
        plot_state_labels_only=False,
    ):
        global state_archive
        my_crs = MY_CRS
        empire = self.gdf
        state_label = self.name
        state_label_font_size = 48


        state_archive[state_label] = empire
        if plot_state_labels_only:
            plot_cities = False
            plot_labels = False

        ax.annotate(
            text=state_label,
            xy=self.state_center(),
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
                    self.annotate_region(region, ax=ax)
                except:
                    raise

        if plot_cities:
            for city in self.cities:
                self.annotate_location(**city, ax=ax)

        # Limit the map to EU and convert to 3857 to improve printing.
        empire = empire.to_crs(my_crs)

        # Draw borders with different colors.
        if plot_geo:
            empire.plot(
                ax=ax, edgecolor="black", facecolor=facecolor2, linewidth=2, alpha=1.0, zorder=ZORDER_REGION_FC2
            )
            empire.plot(
                ax=ax, edgecolor="black", facecolor=facecolor1, linewidth=0, alpha=0.5, zorder=ZORDER_REGION_FC1
            )
        else:
            empire.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0, alpha=0, zorder=ZORDER_REGION_FC0)

        return empire

    def annotate_location(
        self,
        address,
        text=LARGE_CITY,
        fontsize=24,
        fontname="DejaVu Serif",
        ax=plt,
        **kwargs,
    ):
        if text == "●":
            return None
        coords = geolocate(address)
        if not coords:
            log.error(f"Cannot find location: {address}, skipping")
            return None

        log.warning(f"Annotating {address} with {text}, kwargs: {kwargs}")
        tx, ty = self._state.get("translate", [0, 0])
        a11, a22 = self._state.get("scale", [1, 1])
        log.warning(f"Annotating {text} at {coords} with translate {tx}, {ty} and scale {a11}, {a22}")
        annotate_coords(xy=coords, text=text, translate=(tx,ty), scale=(a11,a22), fontname=fontname,
        fontsize=fontsize,ax=ax, **kwargs)



#
# Render maps.
#
from multiprocessing import Manager

manager = Manager()
state_archive = manager.dict()

class Board:
    def __init__(self, name, config, background=False, ax=None) -> None:
        self.config = config
        self.name = name
        if ax is None:
            self.fig, self.ax = get_board()
        if background:
            eu = config.get_europe().to_crs(MY_CRS)
            eu.plot(ax=self.ax, facecolor="lightblue", zorder=ZORDER_BG)

    def render_seas(self):
        seas = self.config.seas
        fontconfig = seas.get("config", {})
        for s in seas.get("items", []):
            x = dict(fontconfig, **s)
            annotate_coords(ax=self.ax, **x)

    def save(self, dpi=92, bbox_inches="tight", transparent=True, **kwargs):
        #cfg = dict(dpi=92, bbox_inches="tight", transparent=True)
        self.fig.savefig(f"/tmp/{self.name}-board-{config.suffix}.png", 
            dpi=dpi, bbox_inches=bbox_inches, transparent=transparent, **kwargs)

    def get_state(self, state_name):
        return State(state_name, self.config)
    def render_links(self):
        # import pdb; pdb.set_trace()
        for link in self.config.links:
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
                line.plot(ax=self.ax, color="black", linewidth=2, linestyle="dashed", zorder=ZORDER_LINKS)


def render_board(countries=COUNTRIES, background=False, plot_cities=True, render_net=True, render_links=False, config=None, **kwargs):
    countries = countries or COUNTRIES
    config = config or Config()

    fig_risk, risk_board = get_board()
    full_board = Board("full", background=True, config=config)
    label_board = Board("label", config=config)
    links_board = Board("links", config=config)
    if background:
        eu = config.get_europe().to_crs(MY_CRS)
        eu.plot(ax=risk_board, facecolor="lightblue")

    with Pool(processes=20) as pool:
        pool.map(
            partial(
                lambda state, **kwargs: State(state, config=config).render(**kwargs),
                ax=full_board.ax,
                plot_geo=True,
                plot_labels=True,
                plot_cities=plot_cities,
                plot_state_labels=False,
            ),
            countries,
        )
        #    pool.map(partial(render_state, ax=label_board, plot_geo=False, plot_labels=True, plot_cities=True,  plot_state_labels=False), countries)
    #    pool.map(partial(render_state, ax=links_board, plot_geo=False, plot_labels=False, plot_cities=False, plot_state_labels=False), countries)

    full_board.render_seas()
    if render_links:
        full_board.render_links()

    label_board.render_links()
    label_board.render_seas()
    label_board.save()

    if render_net:
        add_neighbour_net(risk_board)

    # add_basemap(full_board, crs=str(MY_CRS), )
    full_board.save()


def add_neighbour_net(ax, config=None):
    log.warning("Rendering net.")
    nbr = {}
    df = State(COUNTRIES[0], config=config).gdf
    for x in COUNTRIES[1:]:
        df = df.append(State(x, config=config).gdf)
    df = df.intersects(config.get_europe())
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


def find_region(region_name):
    global state_archive
    for state, gdf in state_archive.items():
        region = gdf[gdf.name == region_name]
        if not region.empty:
            return region
    return None

def save_states(config):
    for c in COUNTRIES:
        f = c.replace("\n", " ")
