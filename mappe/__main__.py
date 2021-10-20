from sys import argv

from . import render_board, test_render_background_masked_ok

countries = argv[1:] if len(argv) > 1 else None
# test_render_background_masked_ok()
render_board(countries)

