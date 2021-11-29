from sys import argv
import click
from . import render_board, test_render_background_masked_ok

@click.command()
@click.argument('countries', nargs=-1)
@click.option('--plot-cities', default=True, help='Board to render')
def main(countries, plot_cities):
    countries = countries or None
    # test_render_background_masked_ok()
    render_board(countries, background=True, plot_cities=plot_cities)

main()