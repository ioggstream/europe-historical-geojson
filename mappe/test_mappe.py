from geopandas.geodataframe import GeoDataFrame
import yaml
from pathlib import Path
from . import (
    COUNTRIES,
    annotate_region,
    get_board,
    get_state,
    maps,
    plot_net,
    prepare_neighbor_net,
    togli_isolette,
    _get_europe
)


def test_save_states():
    for c in COUNTRIES:
        f = c.replace("\n", " ")
        df = get_state(c, cache=False)
        try:
            togli_isolette(df, 0.4)
        except:
            pass
        df.to_file(f"tmp-{f}.geojson", driver="GeoJSON")


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