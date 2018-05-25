import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/bokehjournal")
from journal import Journal

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral6, Reds3, Inferno256, Category10
from bokeh.models import Div, Legend, LinearColorMapper
from bokeh.layouts import column, row, gridplot
import datashader as ds
import datashader.glyphs
import datashader.transfer_functions as t_func
from datashader import reductions
from datashader.core import bypixel

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.morphology import watershed, local_minima
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

GRAPH_WIDTH = 600
GRAPH_HEIGHT = 600

journal = Journal('2018-05-23')
journal.append_h1('Developing A Segmap: Attempt 1')

src_name = 'GDS_deep2_5692'
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

journal.append_paragraph(f'Examinging a segmap around src {src_name}')

grid = [[], []]
for i, band in enumerate('HJVZ'):
    if f'{band}.fits' not in os.listdir():
        big_img = fits.getdata(f'../data/data_5class/{band}.fits')
        small_img = big_img[y-60:y+60, x-60:x+60]
    else:
        small_img = fits.getdata(f'{band}.fits')

    f = figure(title=f'{band} Band Image',
               width=GRAPH_WIDTH,
               height=GRAPH_HEIGHT,
               x_range=[0,120],
               y_range=[0,120])

    f.image(image=[small_img],
            x=0,
            y=0,
            dw=small_img.shape[0],
            dh=small_img.shape[1],
            palette="Greys256")

    grid[i//2].append(f)

journal.append_bokeh(gridplot(grid))


mn = np.load('GDS_deep2_5692-background-Mean.npy')

journal.append_br(num=2)

journal.append_bokeh_img(mn, title='Mean P(pixel=Background)')

journal.append_br(num=2)

journal.append_h3('Algorithm For Segmenting:')
steps = ['Step 1 :Find all local minima using skimage.morphology.local_minima',
         """Step 2: Create a mask based on the background classification,
         where P(pixel=background)<0.3""",
         """Step 3: Assign a unique value greater than 1 to every minima within the mask
         and assign 1 to all other minimas""",
         """Step 4: Apply watershedding algorithm using skimage.morphology.watershed
         where minima greater than 1 identify an candidate and minima equal to 1
         are considered background
         """]

journal.append_list(steps)

local_min = local_minima(mn)
journal.append_bokeh_img(local_min, title='Step 1: Local Minima')

mask = mn < 0.3
journal.append_bokeh_img(mask, title='Step 2: Get Mask P(pixel=Background<0.3)')

masked_min = local_min * mask
count = 0
for i in range(mn.shape[0]):
    for j in range(mn.shape[1]):
        if masked_min[i,j]==1:
            count += 1
            masked_min[i,j] = count

masked_min = masked_min + local_min

journal.append_bokeh_img(masked_min,
                         title='Step 3: Get Mask P(pixel=Background<0.3)')

segmented = watershed(mn, masked_min)
journal.append_bokeh_img(segmented,
                         title='Step 4: Apply Watershedding')

f = figure(title=f'Combined',
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            x_range=[0,120],
            y_range=[0,120])
items = []
for name, data in [('Mean', mn), ('Segmap', segmented)]:
    img_plt = f.image(image=[data],
                      x=0,
                      y=0,
                      dw=small_img.shape[0],
                      dh=small_img.shape[1],
                      color_mapper=LinearColorMapper(palette=Inferno256),
                      legend=name)

    items.append((name, [img_plt]))

f.legend.location = 'top_left'
f.legend.click_policy="hide"

journal.append_bokeh(f)

journal.show(file_name='{}.html'.format(src_name))