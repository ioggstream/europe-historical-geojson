from itertools import product

import pytest
import yaml
from contextily import add_basemap
from contextily import providers as contextily_providers
from matplotlib import pyplot as plt

from mappe import MY_CRS, Board, Config, State

states = ["Ottomano"]
cities = [True]
labels = [True]


@pytest.mark.parametrize(
    "state,plot_cities,plot_labels", product(states, cities, labels)
)
def test_state_(state, plot_cities, plot_labels):
    plot_state(state, plot_cities=plot_cities, plot_labels=plot_labels)


def plot_state(name, plot_cities, plot_labels):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 20)
    state = State(state_name=name, config=Config())
    state.render(ax=ax, plot_cities=plot_cities, plot_labels=plot_labels)
    fig.savefig(f"/tmp/deleteme-{name}.png")


def test_render_state_bg():
    """
    Render a state adding a basemap from contextily.
    """
    config = Config()
    board = Board("state-bg", config)
    state = State("Italia", config)
    state.render(board.ax)
    add_basemap(
        board.ax,
        crs=str(MY_CRS),
        # source=ctx.providers.Esri.WorldPhysical,
        source=contextily_providers.Esri.WorldShadedRelief,
    )
    board.save(dpi=300, transparent=True)


def test_render_background_masked_ok():
    config = Config()
    board = Board("masked-terrain", config)
    diff = config.get_europe().to_crs(MY_CRS)

    # Unveil
    for c in config.states:
        if c in ("UK", "UK2"):
            continue
        diff = diff - State(c, config).get_state().to_crs(MY_CRS).unary_union

    diff.plot(ax=board.ax, color="lightblue")
    add_basemap(
        board.ax,
        crs=str(MY_CRS),
        # source=ctx.providers.Esri.WorldPhysical,
        source=contextily_providers.Esri.WorldShadedRelief,
    )
    board.save(transparent=True)


def test_count_territories():
    d = yaml.safe_load(open("mappe.yaml"))
    assert 42 == len(
        [
            region
            for x in d["maps"].values()
            for region, values in x["regions"].items()
            if not values.get("neutral")
        ]
    )
