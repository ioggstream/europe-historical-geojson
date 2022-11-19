import click

from . import render_board


@click.command()
@click.argument('countries', nargs=-1)
@click.option('--plot-cities', default=False, is_flag=True, help='Whether to plot cities')
@click.option('--render-net', default=False, is_flag=True, help='Whether to render net')
def main(countries, plot_cities, render_net):
    countries = countries or None
    # test_render_background_masked_ok()
    render_board(countries, background=True, plot_cities=plot_cities,
    render_net=render_net)

main()
