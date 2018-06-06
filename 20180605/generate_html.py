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
from bokeh.models import Div, Legend, HoverTool
from bokeh.models.mappers import LinearColorMapper
from bokeh.layouts import column, row, gridplot
import datashader as ds
import datashader.glyphs
import datashader.transfer_functions as t_func
from datashader import reductions
from datashader.core import bypixel

from skimage.morphology import watershed, local_minima

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



# pulled segmap info from 20180518

morphs = ['Spheroid', 'Disk', 'Irregular', 'Point Source', 'Background']
morph_probs = {m:np.load(f'{src_name}-{m}.npy') for m in morphs}

all_vals = np.array([morph_probs[m] for m in morphs])
total = all_vals.sum(axis=0)
all_vals = all_vals / total

size = morph_probs[morphs[0]].shape

cmap = LinearColorMapper(palette='Inferno256', low=0, high=1)
f = figure(x_range=[0, size[1]],
           y_range=[0, size[0]],
           plot_width=650,
           plot_height=GRAPH_HEIGHT,
           toolbar_location=None,
           title='Morphology Probabilities')

items = []
for i, morph in enumerate(morphs):
    img = f.image(image=[all_vals[i,:,:]],
                       x=[0],
                       y=[0],
                       dw=[size[1]],
                       dh=[size[0]],
                       color_mapper=cmap)
    items.append((morph, [img]))

legend = Legend(items=items, location='top_right')
legend.click_policy = 'hide'
legend.spacing = 2
legend.glyph_height = 5*len(items)
legend.label_text_baseline = 'bottom'
f.add_layout(legend, 'right')
journal.append_bokeh(f)

# create segmentation maps based on watershedding from 0528
bkg = all_vals[-1,:,:]
local_min = local_minima(bkg)
mask = bkg<0.3
masked_min = mask * local_min
count = 0
for i in range(bkg.shape[0]):
    for j in range(bkg.shape[1]):
        if masked_min[i,j] == 1:
            count += 1
            masked_min[i,j] = count

masked_min = masked_min + local_min
segmap = watershed(bkg, masked_min)

journal.append_br(num=1)

f = figure(x_range=[0, size[1]],
           y_range=[0, size[0]],
           plot_width=650,
           plot_height=GRAPH_HEIGHT,
           toolbar_location=None,
           title='Segmap Based On Watershedding')

colors = []
for c in Inferno256:
    c = c[1:]
    rgb = tuple([int(c[i:i+2], 16) for i in (0, 2 ,4)] + [0])
    colors.append(rgb)

items = []
for data, name in [(bkg, 'Background'), (segmap, 'Segmap')]:
    cmap = LinearColorMapper(palette='Inferno256',
                             low=data.min(),
                             high=data.max())
    img = f.image(image=[data],
                       x=[0],
                       y=[0],
                       dw=[data.shape[1]],
                       dh=[data.shape[0]],
                       color_mapper=cmap)
    items.append((name, [img]))
legend = Legend(items=items, location='top_right')
legend.click_policy = 'hide'
legend.spacing = 2
legend.glyph_height = 5*len(items)
legend.label_text_baseline = 'bottom'
f.add_layout(legend, 'right')
journal.append_bokeh(f)

src_classifications =[]
for i in range(2, count+1):
    src = np.zeros([4])
    src_pixels = np.where(segmap==i)
    cnt = 0
    for y,x in zip(*src_pixels):
        cnt += 1
        vals = all_vals[:-1,y,x]
        index = list(np.argsort(vals))
        index.reverse()
        for j in range(1):
            src[index[j]] += vals[index[j]]

    src_classifications.append((i, src/src.sum()))

pretty_list = []
for s in src_classifications:
    num = s[0]
    vals = [f'{m}:{round(v, 2)}' for m, v in zip(morphs, s[1])]
    pretty_list.append((num, vals))

for num, vals in pretty_list:
    journal.append_paragraph(f'Source:{num}')
    journal.append_list(vals)

journal.show(file_name='{}.html'.format(src_name))