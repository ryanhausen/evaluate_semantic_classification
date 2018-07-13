import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/bokehjournal")
from journal import Journal

import numpy as np
import pandas as pd
from astropy.io import fits

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral6, Reds3, Inferno256, Category20_20, Greys256, Inferno
from bokeh.models import Div, Legend, HoverTool, ColorBar
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.layouts import column, row, gridplot
from bokeh.colors import RGB
import datashader as ds
import datashader.glyphs
import datashader.transfer_functions as t_func
from datashader import reductions
from datashader.core import bypixel

from skimage.morphology import watershed, local_minima

GRAPH_WIDTH = 600
GRAPH_HEIGHT = 600

journal = Journal('2018-06-26')

title = 'Examining A Larger Piece of the Sky'
journal.append_h1(title)

imgs = [fits.getdata(f'{b}.fits') for b in 'hjvz']

grid = [[], []]
for i, img in enumerate(imgs):

    band = 'HJVZ'[i]
    f = figure(title=f'{band} Band Image',
               width=GRAPH_WIDTH,
               height=GRAPH_HEIGHT,
               x_range=[0,img.shape[1]],
               y_range=[0,img.shape[0]])

    f.image(image=[img],
            x=0,
            y=0,
            dw=img.shape[0],
            dh=img.shape[1],
            palette="Greys256")

    grid[i//2].append(f)

journal.append_bokeh(gridplot(grid))

# pulled segmap info from 20180518

morphs = ['spheroid', 'disk', 'irregular', 'point_source', 'background']
morph_probs = {m:fits.getdata(f'{m}_mean.fits') for m in morphs}
morph_std = {m:fits.getdata(f'{m}_var.fits') for m in morphs}

all_vals = np.array([morph_probs[m] for m in morphs])
total = all_vals.sum(axis=0)
all_vals = all_vals / total

all_var = np.array([morph_std[m] for m in morphs])

size = morph_probs[morphs[0]].shape

cmap = LinearColorMapper(palette='Inferno256', low=0, high=1)
f = figure(x_range=[0, size[1]],
           y_range=[0, size[0]],
           plot_width=700,
           plot_height=GRAPH_HEIGHT,
           toolbar_location=None,
           title='Morphology Probabilities')
f.title.text_font_size = "25px"

items = []
for i, morph in enumerate(morphs):
    img = f.image(image=[all_vals[i,:,:]],
                       x=[0],
                       y=[0],
                       dw=[size[1]],
                       dh=[size[0]],
                       color_mapper=cmap)
    items.append((morph, [img]))

color_bar = ColorBar(color_mapper=cmap,
                     label_standoff=12, border_line_color=None, location=(0,0))

f.add_layout(color_bar, 'right')

legend = Legend(items=items, location='top_right')
legend.click_policy = 'hide'
legend.spacing = 2
legend.glyph_height = 5*len(items)
legend.label_text_baseline = 'bottom'
f.add_layout(legend, 'right')


journal.append_bokeh(f)

low = all_var.min()
high = all_var.max()

cmap = LinearColorMapper(palette='Inferno256', low=low, high=high)
f = figure(x_range=[0, size[1]],
           y_range=[0, size[0]],
           plot_width=700,
           plot_height=GRAPH_HEIGHT,
           toolbar_location=None,
           title='Morphology Variance')
f.title.text_font_size = "25px"

items = []
for i, morph in enumerate(morphs):
    img = f.image(image=[all_var[i,:,:]],
                       x=[0],
                       y=[0],
                       dw=[size[1]],
                       dh=[size[0]],
                       color_mapper=cmap)
    items.append((morph, [img]))

color_bar = ColorBar(color_mapper=cmap,
                     label_standoff=12, border_line_color=None, location=(0,0))

f.add_layout(color_bar, 'right')

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

fits.PrimaryHDU(segmap).writeto('segmap.fits')

journal.append_br(num=1)

f = figure(x_range=[0, size[1]],
           y_range=[0, size[0]],
           plot_width=650,
           plot_height=GRAPH_HEIGHT,
           toolbar_location=None,
           title='Segmap Based On Watershedding')

colors = []
for c in Category20_20:
    c = c[1:]
    rgb = RGB(*tuple([int(c[i:i+2], 16) for i in (0, 2 ,4)] + [0.5]))
    colors.append(rgb)

icolors = []
for c in Category20_20:
    c = c[1:]
    rgb = RGB(*tuple([int(c[i:i+2], 16) for i in (0, 2 ,4)] + [0.5]))
    icolors.append(rgb)



items = []
results = [
    (imgs[0], 'H', Greys256),
    (imgs[1], 'J', Greys256),
    (imgs[2], 'V', Greys256),
    (imgs[3], 'Z', Greys256),
    (bkg, 'Background', Inferno256),
    (segmap, 'Segmap', colors)]
for data, name, palette in results:
    cmap = LinearColorMapper(palette=palette,
                             low=data.min(),
                             high=data.max())
                             
    log_cmap = LogColorMapper(palette=palette)
    img = f.image(image=[data],
                       x=[0],
                       y=[0],
                       #global_alpha=0.5,
                       dw=[data.shape[1]],
                       dh=[data.shape[0]],
                       color_mapper=log_cmap if name in 'HJVZ' else cmap,
                       legend=name)
    items.append((name, [img]))
f.legend.click_policy = 'hide'
# legend = Legend(items=items, location='top_right')
# legend.click_policy = 'hide'
# legend.spacing = 2
# legend.glyph_height = 5*len(items)
# legend.label_text_baseline = 'bottom'
# f.add_layout(legend, 'right')

journal.append_bokeh(f)

journal.show(file_name='{}.html'.format(title))
