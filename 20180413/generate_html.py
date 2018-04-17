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

# ==============================================================================
# For a single object for which we have a human classification:
# - fits file
# - segmentation map
# - tab delimited file with the following columns for all pixels in the segmap
# -- x, y, n, p_classifications
# - produce graphs similar to the first graphs
# ==============================================================================

journal = Journal('2018-04-13')

src_name = 'GDS_deep2_5510'
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
    if f'{band}.fits' not in os.listdir():
        big_img = fits.getdata(f'../data/data_201804/{band}.fits')
        small_img = big_img[y-42:y+42, x-42:x+42]
    else:
        small_img = fits.getdata(f'{band}.fits')

    f = figure(title=f'{band} Band Image',
               width=GRAPH_WIDTH,
               height=GRAPH_HEIGHT,
               x_range=[0,84],
               y_range=[0,84])

    f.image(image=[small_img],
            x=0,
            y=0,
            dw=small_img.shape[0],
            dh=small_img.shape[1],
            palette="Greys256")

    grid[i//2].append(f)

journal.append_bokeh(gridplot(grid))
del big_img
del small_img

journal.append_br(num=2)
journal.append_paragraph('The human classified segmap looks like:')

segmap = fits.getdata(f'../data/data_201804/segmaps/{src_name}_segmap.fits')
f = figure(title=f'{src_name} Segmap',
           width=GRAPH_WIDTH,
           height=GRAPH_HEIGHT,
           x_range=[0, segmap.shape[0]],
           y_range=[0, segmap.shape[1]])
f.image(image=[segmap],
        x=0,
        y=0,
        dw=segmap.shape[0],
        dh=segmap.shape[1],
        palette='Greys256')
journal.append_bokeh(f)
del segmap

journal.append_br(num=2)


adjusted_y = y
adjusted_x = x-600
print(adjusted_x, adjusted_y)

x_start, x_end = adjusted_x-42, adjusted_x+42
y_start, y_end = adjusted_y-42, adjusted_y+42

slices = fits.getdata('../data/slices.fits')             #[num_slices, 40, 40, 1]
predictions = fits.getdata('../data/slices_labels.fits') #[num_slices, 40, 40, 5]

columns = ['y', 'x', 'Spheroid', 'Disk', 'Irregular',
           'Point Source', 'Unknown', 'Background']
if 'points.csv' not in os.listdir():
    roi_ys = np.arange(y_start, y_end, dtype=np.int32)
    roi_xs = np.arange(x_start, x_end, dtype=np.int32)

    print(x, y)
    print(roi_xs, len(roi_xs))
    print(roi_ys, len(roi_ys))

    def get_valid_pixels(slice_idx):
        """
        Input:  slice index
        Output: a list of absolute pixel coords (y,x) and relative coords (i, j)
                if any pixels in the sliceare in the ROI and the ROI pixels are
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

    for slice_idx in range(slices.shape[0]):
        #print(slice_idx/slices.shape[0], end='\r')
        print('row:{}\tcol:{}\t\t\t\t'.format(slice_idx//361, slice_idx%361), end='\r')
        roi_coords = get_valid_pixels(slice_idx)
        if roi_coords:
            with open('valid_slices', 'a') as f:
                print(slice_idx, file=f)
            for abs_coords, rel_coords in roi_coords:
                y, x = rel_coords
                probs = predictions[slice_idx, y, x, :]
                for col, val in zip(columns, list(abs_coords) + list(probs)):
                    data[col].append(val)

    data = pd.DataFrame.from_dict(data)
    data.to_csv('./points.csv')
else:
    data = pd.read_csv('./points.csv')

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
                                   x_range=[0, 150],
                                   y_range=[0, 150])

        agg = canvas.points(data, 'x', 'y', agg=reduction_op)
        img = t_func.shade(agg,
                           cmap=Inferno256,
                           span=[0,0.5] if op_name=='Standard Deviation' else [0,1],
                           how='linear')

        img_plt = f.image_rgba(image=[img.data],
                            x=x_start,
                            y=y_start,
                            dw=x_end-x_start,
                            dh=y_end-y_start,
                            visible=morphology=='Disk')

        items.append((morphology, [img_plt]))

    legend = Legend(items=items,
                    location=(0,250),
                    click_policy='hide',
                    )

    f.add_layout(legend, place='right')

    probs_grid[i//2].append(f)

journal.append_bokeh(gridplot(probs_grid))

journal.show()

