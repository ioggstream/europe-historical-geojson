from pathlib import Path

import geopandas as gpd
import yaml
from matplotlib import pyplot as plt
from utils import Config, annotate_coords

from . import (
    COUNTRIES,
    MY_CRS,
    _get_europe,
    add_basemap,
    annotate_region,
    ctx,
    get_board,
    get_historical_borders,
    get_state,
    intersect,
    maps,
    plot_net,
    prepare_neighbor_net,
)


def test_render_background_masked_ok():
    diff = Config().get_europe().to_crs(MY_CRS)
    for c in COUNTRIES:
        diff = diff - get_state(c).to_crs(MY_CRS).unary_union

    fig, ax = get_board()
    diff.plot(ax=ax, color="lightblue")
    add_basemap(
        ax,
        crs=str(MY_CRS),
        # source=ctx.providers.Esri.WorldPhysical,
        source=ctx.providers.Esri.WorldShadedRelief,
    )
    fig.savefig("masked-terrain-board.png", dpi=300, transparent=True)


def test_unite_maps():
    fig, ax = get_board()
    mask = gpd.read_file(open("germany-1914-boundaries.geojson")).set_crs(
        EPSG_4326_WGS84
    )
    de = get_area("PL")
    de = de.append(get_area("DE"))
    de = intersect(de, mask)
    de.plot(ax=ax)
    print(de)
    fig.savefig("/tmp/test-intersect.png")


def test_get_historical_borders():
    borders = get_historical_borders(
        {"country-borders": ["germany-1914-boundaries.geojson", "BE"]}
    )
    plt.plot(borders[0].exterior.xy[0], borders[0].exterior.xy[1])
    plt.show()


def test_render_background_ok():
    fig, ax = get_board()
    eu = _get_europe().to_crs(MY_CRS)
    eu.plot(ax=ax, color="none")

    add_basemap(ax, crs=str(MY_CRS), source=ctx.providers.Esri.WorldPhysical)
    fig.savefig("/tmp/terrain-board.png", dpi=300, transparent=True)


def test_generate_net():
    fig, ax = get_board()
    df = get_state(COUNTRIES[0])
    for x in COUNTRIES[1:]:
        df = df.append(get_state(x))
    df = intersect(df, _get_europe())
    df.plot(ax=ax)
    nbr = {}
    prepare_neighbor_net(df, nbr)
    plot_net(nbr, ax)

    for region_id, values in nbr.items():
        annotate_coords(
            values["coords"],
            f"w: {values['count']}",
            textcoords="offset points",
            xytext=(-20, -20),
            ax=ax,
            color="black",
        )

    fig.savefig("./tmp-nbr_net.png")


def test_stats():
    m = maps()
    countries = len(m)
    regions = sum(len(v["regions"]) for v in m.values())
    print(f"{countries} countries, {regions} regions")


def test_nbr_net():
    fig, ax = get_board()
    nbr = {}
    df = get_state("France")
    prepare_neighbor_net(df, nbr)
    plot_net(nbr, ax)
    df.plot(ax=ax, color="yellow")
    fig.savefig("/tmp/test_nbr_net.png")


def test_full_net():
    gdf = get_state(COUNTRIES[1])
    for c in COUNTRIES[2:]:
        gdf = gdf.append(get_state(c))
    nbr = {}
    prepare_neighbor_net(gdf, nbr)
    print(yaml.safe_dump(nbr))
    Path("/tmp/test_nbr_net.yaml").write_text(yaml.safe_dump(nbr))
    fig, ax = get_board()
    plot_net(nbr, ax)
    gdf.plot(ax=ax, color="yellow")
    fig.savefig("/tmp/test_nbr_net.png")


def test_annotate_region():
    gdf = get_state("France")
    region_id = gdf.name[0]
    region = gdf[gdf.name == region_id]
    annotate_region("France", region)
    raise NotImplementedError


def test_render_state():
    fig, ax = get_board()
    render_state("Italia", ax=ax)
    fig.savefig("/tmp/test-render-state.png", dpi=300)


def test_render_labels_ok():
    fig_label, label_board = get_board()
    with Pool(processes=20) as pool:
        pool.map(
            partial(
                render_state,
                ax=label_board,
                plot_geo=False,
                plot_labels=True,
                plot_cities=False,
                plot_state_labels=False,
            ),
            COUNTRIES,
        )
    # render_state(state_label="Italia", ax=label_board, plot_geo=False, plot_labels=True)
    fig_label.savefig("label-board.eps", dpi=300, transparent=True, format="eps")


from functools import partial
from multiprocessing import Pool

from . import render_state


def test_render_cities_ok():
    fig_label, label_board = get_board()
    with Pool(processes=20) as pool:
        pool.map(
            partial(
                render_state,
                ax=label_board,
                plot_geo=False,
                plot_labels=False,
                plot_cities=True,
                plot_state_labels=False,
            ),
            COUNTRIES,
        )
    fig_label.savefig("cities-board.eps", dpi=300, transparent=True, format="eps")


def test_render_state_labels_ok():
    fig_label, label_board = get_board()
    with Pool(processes=20) as pool:
        pool.map(
            partial(
                render_state,
                ax=label_board,
                plot_geo=False,
                plot_labels=False,
                plot_cities=False,
                plot_state_labels=True,
                plot_state_labels_only=True,
            ),
            COUNTRIES,
        )
    fig_label.savefig("state_labels-board.eps", dpi=300, transparent=True, format="eps")
