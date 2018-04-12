import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/bokehjournal")
from journal import Journal

import numpy as np
from astropy.io import fits

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral6, Reds3, Inferno256, Category10
from bokeh.models import Div, Legend
from bokeh.layouts import column, row, gridplot
import datashader as ds
import datashader.glyphs
import datashader.transfer_functions as t_func
from datashader import reductions
from datashader.core import bypixel

GRAPH_WIDTH = 800
GRAPH_HEIGHT = 600

# ==============================================================================
# For a single object for which we have a human classification:
# - fits file
# - segmentation map
# - tab delimited file with the following columns for all pixels in the segmap
# -- x, y, n, p_classifications
# - produce graphs similar to the first graphs
# ==============================================================================

journal = Journal('2018-04-13')

src_name = 'GDS_deep2_5961'
journal.append_h1(f'Examining {src_name}')


with open('../data/data_201804/sources_with_locations_and_labels', 'r') as f:
    for l in f:
        vals = l.split(',')
        if vals[0]==src_name:
            x = int(float(vals[1]))
            y = int(float(vals[2]))
            sph = float(vals[3])
            dk = float(vals[4])
            irr = float(vals[5])
            ps = float(vals[6])

journal.append_paragraph(f'{src_name} is has a human classification of:')

morphs = ['Spheroid', 'Disk', 'Irregular', 'Point Source']
vals = [sph, dk, irr, ps]
classifications = [f'{m}:{v}' for m, v in zip(morphs, vals)]
journal.append_list(classifications)

grid = [[], []]
for i, band in enumerate('HJVZ'):
    big_img = fits.getdata(f'..data/data_201804/{band}.fits')
    small_img = big_img[y-40:y+40, x-40:x+40]
    f = figure(title=band, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)
    f.image(image=[small_img],
            x=x_start,
            y=y_start,
            dw=width,
            dh=height,
            palette="Greys256")





journal.show()

