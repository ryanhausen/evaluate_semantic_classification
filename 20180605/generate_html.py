import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/bokehjournal")
from journal import Journal

import numpy as np
import pandas as pd
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

GRAPH_WIDTH = 600
GRAPH_HEIGHT = 600


journal = Journal('2018-06-05')

src_name = 'GDS_deep2_5692'
journal.append_h1(f'Examining {src_name}')

with open('../data/data_5class/sources_with_locations_and_labels', 'r') as f:
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

# pulled segmap info from 20180518

morphs = ['Spheroid', 'Disk', 'Irregular', 'Point Source', 'Background']
morph_probs = {m:np.load(f'{src_name}-{m}.npy') for m in morphs}
size = morph_probs[morphs[0]].shape

f = figure(x_range=[0, size[1]],
           y_range=[0, size[0]],
           width=GRAPH_WIDTH,
           height=GRAPH_HEIGHT,
           toolbar_location='above',
           title='Morphology Probabilities')
legend_items = []
for morph in morphs:
    img = f.image_rgba(image=morph_probs[morph],
                       x=0,
                       y=0,
                       dw=size[1],
                       dh=size[0])

    legend_items.append((morph, img))

legend = Legend(items=legend_items,
                location=(0,250),
                click_policy='hide')
f.add_layout(legend, place='right')
journal.append_bokeh(f)


journal.show(file_name='{}.html'.format(src_name))