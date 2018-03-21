import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Spectral6
from bokeh.models import Div, ColorBar, LinearColorMapper
from bokeh.layouts import column, row

# slices = [num_slices, 40, 40, 6]
slices = fits.getdata('slices_labels.fits')
in_slices = fits.getdata('slices.fits')
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

# all_plots = [Div(text='<h1 style="text-align:center">Top 2 Classifications</h1>', width=1200)]
# bins = [100, 25, 10, 4, 4]
# px_counts = ['2', '4', '10', '20', '40']
# px_counts = ['40', '20', '10', '4', '2']

# output_file("pixel_dist.html")
# for k, b in zip(px_counts, bins):

#     title = '<h3 style="text-align:center">{}x{} Classification Distributions for: ({},{})</h3>'.format(k,k,x,y)
#     f_pdf = figure(title='PDF',
#                    x_axis_label='P(pixel=class)',
#                    y_axis_label='Normalized Pixel Count',
#                    x_range=[0,1.1], 
#                    y_range=[0,1.1],
#                    width=600)
                   
#     f_cdf = figure(title='CDF',
#                    x_axis_label='P(pixel=class)',
#                    y_axis_label='Normalized Pixel Count',
#                    x_range=[0,1.1], 
#                    y_range=[0,1.1],
#                    width=600)

#     for c, n, color in zip(colls[k], names, Spectral6):
#         if len(c) > 0:
#             _y, _x = np.histogram(c, bins=len(c) if len(c) < 16 else b)
#             _x = _x[:-1] + np.diff(_x)/2
#             _y = _y/_y.sum()
            
#             #f_pdf.line(_x, _y, legend=n)
#             #f_cdf.line(_x, _y.cumsum(), legend=n)
#             if len(c)>1:
#                 f_pdf.line(_x, _y, legend=n, line_color=color, line_width=4)
#                 f_cdf.line(_x, _y.cumsum(), legend=n, line_color=color, line_width=4)
#             else:
#                 f_pdf.circle(_x, _y, legend=n, line_color=color, line_width=4)
#                 f_cdf.circle(_x, _y.cumsum(), legend=n, line_color=color, line_width=4)

#     f_cdf.legend.location = 'top_right'
#     f_cdf.legend.click_policy = 'mute'
#     f_pdf.legend.location = 'top_right'
#     f_pdf.legend.click_policy = 'mute'
    
    
#     cmb = row(f_pdf, f_cdf)
#     f = column(Div(text=title, width=1200), cmb)
#     all_plots.append(f)

# f_all = column(*all_plots)    
# show(f_all)



output_file('spatial_dist.html', title='Spatial Distribution')
f_spatial = figure(title='Disk Over Pixels',
                   x_range=[0,40],
                   y_range=[0,40])

cmap = LinearColorMapper(palette="Inferno256", 
                         low=spatial_dist.min(), 
                         high=spatial_dist.max())

f_spatial.image(image=[spatial_dist], x=0, y=0, dw=40, dh=40, color_mapper=cmap)
color_bar = ColorBar(color_mapper=cmap,
                     label_standoff=12, location=(0,0))
f_spatial.add_layout(color_bar, 'right')
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
