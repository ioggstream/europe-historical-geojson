from . import COUNTRIES, get_board, get_state, plot_net, prepare_neighbor_net, togli_isolette, maps


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
    regions = sum(len(v['regions']) for v in m.values())
    print(f"{countries} countries, {regions} regions")




def test_nbr_net():
    fig, ax = get_board()
    nbr = {}
    df = get_state("France")
    prepare_neighbor_net(df, nbr)
    plot_net(nbr, ax)
    df.plot(ax=ax, color="yellow")
    fig.savefig("/tmp/test_nbr_net.png")
