import click

from . import Config, State, render_board


@click.command()
@click.argument('countries', nargs=-1)
@click.option('--plot-cities', default=False, is_flag=True, help='Whether to plot cities')
@click.option('--render-net', default=False, is_flag=True, help='Whether to render net')
@click.option('--save', default=False, is_flag=True, help='Whether to save state boundaries')
def main(countries, plot_cities, render_net, save):
    config = Config()
    countries = countries or config.maps.keys()

    if save:
        for c in countries:
            state = State(c, config)
            state.save()
        return
    # test_render_background_masked_ok()
    render_board(countries, background=True, plot_cities=plot_cities,
    render_net=render_net, config=config)

main()
