import os
import sys
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral6, Inferno256, Category10
from bokeh.models import Div, Legend
from bokeh.layouts import column, row
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

data_reductions = [
    ('Mean', reductions.mean),
    ('Max', reductions.max),
    ('Standard Deviation', reductions.std)
]

for op_name, op in data_reductions:
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

    outputs.append(f)



# ==============================================================================


# ==============================================================================
# Plot PDFs for all pixels
f = figure(x_range=[-0.1, 1.1],
        y_range=[-0.1, 1.1],
        width=GRAPH_WIDTH,
        height=GRAPH_HEIGHT,
        toolbar_location='above',
        title='PDFs For all Pixels')

for morphology, color in zip(columns[2:], Spectral6):
    results, edges = np.histogram(data[morphology].values, bins=100)
    x = edges[:-1] + np.diff(edges)/2
    y = results / results.sum()
    width = edges[1]-edges[0]
    f.line(x,y, line_color=color, line_alpha=0.5)

outputs.append(f)



# ==============================================================================

output = column(outputs)
show(output)
sys.exit(0)








# pixel in interest
y, x = 60, 102

#row = -1
#for i in range(slices.shape[0]):
    #climb rows
#    if i%361==0:
#        row += 1
#        if row==(y-40):
#            #made it
#            print(i)
#            sys.exit(0)
#            plt.title(row)
#            plt.imshow(in_slices[i,:,:], cmap='gray', origin='lower')
#            plt.show()

# row 60 starts at 7220
# col 102 starts at 7220+102

START_SLICE = 7220+(x-40)


colls = {
        '40':[[], [], [], [], [], []],
        '20':[[], [], [], [], [], []],
        '10':[[], [], [], [], [], []],
        '4':[[], [], [], [], [], []],
        '2':[[], [], [], [], [], []]
        }
names = ['Spheroid', 'Disk', 'Irregular', 'Point Source', 'Uknown', 'Background']

spatial_dist = np.zeros([40,40])

for i in range(40):
    for j in range(40):
        idx_y, idx_x = -(i+1), -(j+1)
        rec_x, rec_y = (40+idx_x), (40+idx_y)
        singl_slice = slices[START_SLICE,:,:,:]
        prob_vector = slices[START_SLICE,idx_y,idx_x,:]

        args = np.argsort(-prob_vector)
        vals = args[:1]
        spatial_dist[idx_y, idx_x] = prob_vector[1]

        arg_count = 0
        # inner 40x40
        for p, coll in zip(prob_vector, colls['40']):
            if arg_count in vals:
                coll.append(p)
            arg_count += 1

        arg_count = 0
        # inner 20x20
        if (i < 30 and i > 9) and (j < 30 and j > 9):
            for p, coll in zip(prob_vector, colls['20']):
                if arg_count in vals:
                    coll.append(p)
                arg_count += 1

        arg_count = 0
        # inner 10x10
        if (i < 25 and i > 14) and (j < 25 and  j > 14):
            for p, coll in zip(prob_vector, colls['10']):
                if arg_count in vals:
                    coll.append(p)
                arg_count += 1

        arg_count = 0
        # inner 4x4
        if (i < 22 and i > 17) and (j < 22 and j > 17):
            for p, coll in zip(prob_vector, colls['4']):
                if arg_count in vals:
                    coll.append(p)
                arg_count += 1

        arg_count = 0
        # inner 2x2
        if (i < 21 and i > 18) and (j < 21 and j > 18):
            for p, coll in zip(prob_vector, colls['2']):
                if arg_count in vals:
                    coll.append(p)
                arg_count += 1


        START_SLICE += 1
    START_SLICE += (361-40)

all_plots = [Div(text='<h1 style="text-align:center">Top 2 Classifications</h1>', width=1200)]
bins = [100, 25, 10, 4, 4]
px_counts = ['2', '4', '10', '20', '40']
px_counts = ['40', '20', '10', '4', '2']

output_file("pixel_dist.html")
for k, b in zip(px_counts, bins):

    title = '<h3 style="text-align:center">{}x{} Classification Distributions for: ({},{})</h3>'.format(k,k,x,y)
    f_pdf = figure(title='PDF',
                   x_axis_label='P(pixel=class)',
                   y_axis_label='Normalized Pixel Count',
                   x_range=[0,1.1],
                   y_range=[0,1.1],
                   width=600)

    f_cdf = figure(title='CDF',
                   x_axis_label='P(pixel=class)',
                   y_axis_label='Normalized Pixel Count',
                   x_range=[0,1.1],
                   y_range=[0,1.1],
                   width=600)

    for c, n, color in zip(colls[k], names, Spectral6):
        if len(c) > 0:
            _y, _x = np.histogram(c, bins=len(c) if len(c) < 16 else b)
            _x = _x[:-1] + np.diff(_x)/2
            _y = _y/_y.sum()

            #f_pdf.line(_x, _y, legend=n)
            #f_cdf.line(_x, _y.cumsum(), legend=n)
            if len(c)>1:
                f_pdf.line(_x, _y, legend=n, line_color=color, line_width=4)
                f_cdf.line(_x, _y.cumsum(), legend=n, line_color=color, line_width=4)
            else:
                f_pdf.circle(_x, _y, legend=n, line_color=color, line_width=4)
                f_cdf.circle(_x, _y.cumsum(), legend=n, line_color=color, line_width=4)

    f_cdf.legend.location = 'top_right'
    f_cdf.legend.click_policy = 'mute'
    f_pdf.legend.location = 'top_right'
    f_pdf.legend.click_policy = 'mute'


    cmb = row(f_pdf, f_cdf)
    f = column(Div(text=title, width=1200), cmb)
    all_plots.append(f)

f_all = column(*all_plots)
show(f_all)

print(spatial_dist.shape)
output_file('spatial_dist.html', title='Spatial Distribution')
f_spatial = figure(title='Disk Over Pixels',
                   x_range=[0,40],
                   y_range=[0,40])
f_spatial.image(image=[spatial_dist], x=0, y=0, dw=40, dh=40, palette="Spectral11")
show(f_spatial)


#    f, (ax_pdf, ax_cdf) = plt.subplots(nrows=1,ncols=2)
#    for axes in [ax_pdf, ax_cdf]:
#        axes.set_xlim(0,1)
#        axes.set_xlabel('P(pixel=class')
#        axes.set_ylabel('Normalized Pixel Count')
#    ax_pdf.set_title('PDF')
#    ax_cdf.set_title('CDF')
#
#    plt.suptitle('{}x{} Classification Distributions for: ({},{})'.format(k,k,x,y))
#
#    for c, n in zip(colls[k], names):
#        _y, _x = np.histogram(c, bins=b)
#        _x = _x[:-1] + np.diff(_x)/2
#        _y = _y/_y.sum()
#        ax_pdf.plot(_x, _y, label=n)
#        ax_cdf.plot(_x, _y.cumsum(), label=n)
#    plt.legend()

#plt.show()

sys.exit(0)
