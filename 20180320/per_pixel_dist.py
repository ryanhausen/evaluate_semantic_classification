import os
import sys
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as clr
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

# ==============================================================================
# Get all classifications for all pixels in ROI
# Use distributions to decide on a segmap
# plot distributions for each class for all pixels in segmap
# plot per pixel classifications on image using mean of the pdf
# ==============================================================================
GRAPH_WIDTH = 800
GRAPH_HEIGHT = 600
PixelProb = namedtuple('PixelProb', ['y', 'x', 'sph', 'dsk', 'irr', 'ps', 'unk', 'bkg'])

output_file('output.html')
outputs = [Div(text='<h1 style="text-align:center">Utilizing The Inner 10x10 Prediction Area</h1>')]

predictions = fits.getdata('../data/slices_labels.fits') #[num_slices, 40, 40, 6]
inputs = fits.getdata('../data/slices.fits')             #[num_slices, 40, 40, 1]
candels = fits.getdata('../data/candels_400x400.fits')   #[400,400]

# ==============================================================================
# Display ROI
# ROI:
# Y: [21,96)
# X: [60,140)
y_start, y_end = 21, 96
height = y_end-y_start

x_start, x_end = 60, 140
width = x_end-x_start

roi = candels[y_start:y_end,x_start:x_end]

f = figure(title='Region of Interest',
           x_range=[x_start, x_end],
           y_range=[y_start, y_end],
           width=GRAPH_WIDTH,
           height=GRAPH_HEIGHT,
           toolbar_location='above')
f.image(image=[roi], x=x_start, y=y_start, dw=width, dh=height, palette="Greys256")
outputs.append(f)
# ==============================================================================


# ==============================================================================
# Extract probabilites
# inner 10x10 is from [15,25)

columns = ['y', 'x', 'Spheroid', 'Disk', 'Irregular', 'Point Source', 'Unknown', 'Background']
if 'points.csv' not in os.listdir('.'):
    roi_ys = np.arange(y_start, y_end, dtype=np.int32)
    roi_xs = np.arange(x_start, x_end, dtype=np.int32)

    def get_valid_pixels(slice_idx):
        """
        Input:  slice index
        Output: a list of absolute pixel coords (y,x) and relative coords (i, j) if
                any pixels in the sliceare in the ROI and the ROI pixels are
                classified using the inner10x10 portion of the classifier.
                None otherwise.
        """

        # the absolute position of (0,0) for this slice
        row = slice_idx // 361
        col = slice_idx % 361

        # get the absolute indicies of the inner 10x10 pixels for this slice
        ys = np.arange(row+15, row+25)
        xs = np.arange(col+15, col+25)

        roi_pixels = []
        for i in range(10):
            for j in range(10):
                if (ys[i] in roi_ys) and (xs[j] in roi_xs):
                    abs_coords = (ys[i], xs[j])
                    rel_coords = (15+i, 15+j)
                    roi_pixels.append((abs_coords, rel_coords))

        return roi_pixels if len(roi_pixels) > 0 else None

    data = { c:[] for c in columns}

    for slice_idx in range(inputs.shape[0]):
        print(slice_idx/inputs.shape[0], end='\r')
        roi_coords = get_valid_pixels(slice_idx)
        if roi_coords:
            for abs_coords, rel_coords in roi_coords:
                y, x = rel_coords
                probs = predictions[slice_idx, y, x, :]
                for col, val in zip(columns, list(abs_coords) + list(probs)):
                    data[col].append(val)

    data = pd.DataFrame.from_dict(data)
    data.to_csv('./points.csv')
else:
    data = pd.read_csv('./points.csv')
# ==============================================================================


# ==============================================================================
# Display a segmap based on probablilites
# https://anaconda.org/jbednar/pipeline/notebook

probs_grid = [[], []]

data_reductions = [
    ('Mean', reductions.mean),
    ('Max', reductions.max),
    ('Standard Deviation', reductions.std)
]

for i, (op_name, op) in enumerate(data_reductions):
    f = figure(x_range=[x_start, x_end],
            y_range=[y_start, y_end],
            width=GRAPH_WIDTH,
            height=GRAPH_HEIGHT,
            toolbar_location='above',
            title=f'{op_name} Probabilities Per Class')
    items = []
    for morphology in columns[2:]:
        reduction_op = op(morphology)

        canvas = datashader.Canvas(plot_width=x_end-x_start,
                                plot_height=y_end-y_start,
                                x_range=[x_start, x_end],
                                y_range=[y_start, y_end])

        agg = canvas.points(data, 'x', 'y', agg=reduction_op)
        img = t_func.shade(agg,
                           cmap=Inferno256,
                           span=[0,0.5] if op_name=='Standard Deviation' else [0,1],
                           how='linear')

        img_plt = f.image_rgba(image=[img.data],
                            x=x_start,
                            y=y_start,
                            dw=width,
                            dh=height,
                            visible=morphology=='Disk')

        items.append((morphology, [img_plt]))

    legend = Legend(items=items,
                    location=(0,250),
                    click_policy='hide',
                    )

    f.add_layout(legend, place='right')

    probs_grid[i//2].append(f)

# mean * (1-std)
# f = figure(x_range=[x_start, x_end],
#            y_range=[y_start, y_end],
#            width=GRAPH_WIDTH,
#            height=GRAPH_HEIGHT,
#            toolbar_location='above',
#            title=f'Mean * (1-Std)  Probabilities Per Class ')

# items = []
# for morphology in columns[2:]:
#     reduction_op = reductions.summary(m=reductions.mean(morphology), s=reductions.std(morphology))

#     canvas = datashader.Canvas(plot_width=x_end-x_start,
#                             plot_height=y_end-y_start,
#                             x_range=[x_start, x_end],
#                             y_range=[y_start, y_end])

#     agg = canvas.points(data, 'x', 'y', agg=reduction_op)
#     img = t_func.shade(agg.m * np.subtract(1.0, agg.s),
#                         cmap=Inferno256,
#                         span=[0,1],
#                         how='linear')

#     img_plt = f.image_rgba(image=[img.data],
#                         x=x_start,
#                         y=y_start,
#                         dw=width,
#                         dh=height,
#                         visible=morphology=='Disk')

#     items.append((morphology, [img_plt]))

# legend = Legend(items=items,
#                 location=(0,250),
#                 click_policy='hide',
#                 )

# f.add_layout(legend, place='right')

# outputs.append(f)


f = figure(title='Segmap based mean P(c!=Background)>=P(c==Background)',
           x_range=[x_start, x_end],
           y_range=[y_start, y_end],
           width=GRAPH_WIDTH,
           height=GRAPH_HEIGHT,
           toolbar_location='above')

items = []
for morphology in columns[2:-1]:
    reduction_op = reductions.summary(not_bkg=reductions.mean(morphology),
                                      bkg=reductions.mean('Background'))

    canvas = datashader.Canvas(plot_width=x_end-x_start,
                               plot_height=y_end-y_start,
                               x_range=[x_start, x_end],
                               y_range=[y_start, y_end])

    agg = canvas.points(data, 'x', 'y', agg=reduction_op)
    img = t_func.shade(np.greater_equal(agg.not_bkg,agg.bkg),
                       #cmap=Reds3,
                       span=[0,1],
                       how='linear')

    img_plt = f.image_rgba(image=[img.data],
                        x=x_start,
                        y=y_start,
                        dw=width,
                        dh=height,
                        visible=morphology=='Disk')

    items.append((morphology, [img_plt]))

legend = Legend(items=items,
                location=(0,250),
                click_policy='hide',
                )

f.add_layout(legend, place='right')

probs_grid[1].append(f)
outputs.append(gridplot(probs_grid))

def get_color(color):
    cmap = cm.get_cmap(color)
    return [clr.to_hex(m) for m in cmap(np.arange(cmap.N))]
# Merge as colors
reds = get_color('Reds')
greens = get_color('Greens')
blues = get_color('Blues')

f = figure(title='R=Spheroid, G=Disk, B=Point Source',
           x_range=[x_start, x_end],
           y_range=[y_start, y_end],
           width=GRAPH_WIDTH,
           height=GRAPH_HEIGHT,
           toolbar_location='above')

reduction_op = reductions.summary(sph=reductions.mean('Spheroid'),
                                  dk=reductions.mean('Disk'),
                                  ps=reductions.mean('Point Source'),
                                  bkg=reductions.mean('Background'))

canvas = datashader.Canvas(plot_width=x_end-x_start,
                           plot_height=y_end-y_start,
                           x_range=[x_start, x_end],
                           y_range=[y_start, y_end])

agg = canvas.points(data, 'x', 'y', agg=reduction_op)

imgs = []
for morph, color in [(agg.sph, 'red'),(agg.dk, 'green'), (agg.ps, 'blue'), (agg.bkg, 'black')]:
    imgs.append(t_func.shade(morph, cmap=['white', color], span=[0, 1], how='linear'))

img = t_func.stack(*imgs, how='add')
#img = t_func.set_background(img, color='black')

img_plt = f.image_rgba(image=[img.data],
                       x=x_start,
                       y=y_start,
                       dw=width,
                       dh=height)


outputs.append(f)



# ==============================================================================

# ==============================================================================
# plot in-segmap PDFs
pdf_cdf = []
f = figure(x_range=[-0.1, 1.1],
           y_axis_type='log',
           width=GRAPH_WIDTH,
           height=GRAPH_HEIGHT,
           toolbar_location='above',
           title='PDFs For all Pixels')

for morphology, color in zip(columns[2:], Spectral6):
    reduction_op = reductions.summary(not_bkg=reductions.mean(morphology),
                                      bkg=reductions.mean('Background'))
    canvas = datashader.Canvas(plot_width=x_end-x_start,
                               plot_height=y_end-y_start,
                               x_range=[x_start, x_end],
                               y_range=[y_start, y_end])

    agg = canvas.points(data, 'x', 'y', agg=reduction_op)

    mask = np.greater_equal(agg.not_bkg, agg.bkg)
    print(mask.where(mask==True, drop=True).x.data)


for morphology, color in zip(columns[2:], Spectral6):
    results, edges = np.histogram(data[morphology].values, bins=1000)
    x = edges[:-1] + np.diff(edges)/2
    y = results / results.sum()
    width = edges[1]-edges[0]
    f.line(x,y, line_color=color,
                line_alpha=0.75,
                line_width=4,
                legend=morphology)

f.legend.location = 'bottom_right'
f.legend.click_policy = 'hide'
pdf_cdf.append(f)

 # plot out-segmap PDFs




# ==============================================================================



# ==============================================================================
# Plot PDFs for all pixels
pdf_cdf = []
f = figure(x_range=[-0.1, 1.1],
        y_axis_type='log',
        width=GRAPH_WIDTH,
        height=GRAPH_HEIGHT,
        toolbar_location='above',
        title='PDFs For all Pixels')

for morphology, color in zip(columns[2:], Spectral6):
    results, edges = np.histogram(data[morphology].values, bins=1000)
    x = edges[:-1] + np.diff(edges)/2
    y = results / results.sum()
    width = edges[1]-edges[0]
    f.line(x,y, line_color=color,
                line_alpha=0.75,
                line_width=4,
                legend=morphology)

f.legend.location = 'bottom_right'
f.legend.click_policy = 'hide'
pdf_cdf.append(f)

# Plot CDFs for all pixels
f = figure(x_range=[-0.1, 1.1],
           y_axis_type='log',
           width=GRAPH_WIDTH,
           height=GRAPH_HEIGHT,
           toolbar_location='above',
           title='CDFs For all Pixels')

for morphology, color in zip(columns[2:], Spectral6):
    results, edges = np.histogram(data[morphology].values, bins=1000)
    x = edges[:-1] + np.diff(edges)/2
    y = np.cumsum(results / results.sum())
    width = edges[1]-edges[0]
    f.line(x,y, line_color=color,
                line_alpha=0.75,
                line_width=4,
                legend=morphology)

f.legend.location = 'bottom_right'
f.legend.click_policy = 'hide'
pdf_cdf.append(f)
pdf_cdf = row(pdf_cdf)

outputs.append(pdf_cdf)
# ==============================================================================

output = column(outputs)
show(output)
sys.exit(0)