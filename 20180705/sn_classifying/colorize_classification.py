import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits


# RED = np.array([255, 0, 0, 0], dtype=np.uint8)
# GREEN = np.array([0, 255, 0, 0], dtype=np.uint8)
# BLUE = np.array([0, 0, 255, 0], dtype=np.uint8)
# YELLOW = np.array([255, 255, 0, 0], dtype=np.uint8)

# RED = np.array([255, 0, 0], dtype=np.uint8)
# GREEN = np.array([0, 255, 0], dtype=np.uint8)
# BLUE = np.array([0, 0, 255], dtype=np.uint8)
# YELLOW = np.array([255, 255, 0], dtype=np.uint8)

RED = np.array([1, 0, 0, 1])
GREEN = np.array([0, 1, 0, 1])
BLUE = np.array([0, 0, 1, 1])
YELLOW = np.array([1, 1, 0, 1])



def colorize(bkg, sph, dk, irr, ps, color=4, variance_weighting=True):
    if color == 4:
        return four_color(bkg, sph, dk, irr, ps, variance_weighting)

    bkg_m, bkg_v = bkg
    sph_m, sph_v = sph
    dk_m, dk_v = dk
    irr_m, irr_v = irr
    ps_m, ps_v = ps

    in_y, in_x = bkg_m.shape
    color_img = np.zeros([in_y, in_x, 4], dtype=np.float32)


    for i in range(in_y):
        for j in range(in_x):
            b = bkg_m[i,j]
            s = sph_m[i,j]
            d = dk_m[i,j]
            ir = irr_m[i,j]
            p = ps_m[i,j]

            if variance_weighting:
                safe_var = lambda v: v if v != 0 else 1
                #b *= (1 / safe_var(bkg_v[i,j]))
                s *= (1 / safe_var(sph_v[i,j]))
                d *= (1 / safe_var(dk_v[i,j]))
                ir *= (1 / safe_var(irr_v[i,j]))
                p *= (1 / safe_var(ps_v[i,j]))

                norm_val = sum([s, d, ir, p])
                #b /= safe_var(norm_val)
                s /= safe_var(norm_val)
                d /= safe_var(norm_val)
                ir /= safe_var(norm_val)
                p /= safe_var(norm_val)



            vals = np.array([s,d,ir,p])
            colors = np.array([RED, BLUE, GREEN, YELLOW])

            top_k = np.argsort(-vals)

            for k in range(color):
                color_img[i,j,:] += vals[top_k[k]] * colors[top_k[k]]

            color_img[i,j,-1] = 1-b

    return color_img



def four_color(bkg, sph, dk, irr, ps, variance_weighting=True):

    bkg_m, bkg_v = bkg
    sph_m, sph_v = sph
    dk_m, dk_v = dk
    irr_m, irr_v = irr
    ps_m, ps_v = ps

    in_y, in_x = bkg_m.shape
    color_img = np.zeros([in_y*2, in_x*2, 4], dtype=np.float32)

    for i in range(in_y):
        for j in range(in_x):
            b = bkg_m[i,j]
            s = sph_m[i,j]
            d = dk_m[i,j]
            ir = irr_m[i,j]
            p = ps_m[i,j]


            if variance_weighting:
                safe_var = lambda v: v if v != 0 else 1
                b *= (1 / safe_var(bkg_v[i,j]))
                s *= (1 / safe_var(sph_v[i,j]))
                d *= (1 / safe_var(dk_v[i,j]))
                ir *= (1 / safe_var(irr_v[i,j]))
                p *= (1 / safe_var(ps_v[i,j]))

                norm_val = sum([s, d, ir, p])
                b /= safe_var(norm_val)
                s /= safe_var(norm_val)
                d /= safe_var(norm_val)
                ir /= safe_var(norm_val)
                p /= safe_var(norm_val)




            _i, _j = i*2, j*2
            #color_img[_i, _j, :] = (RED * s).astype(np.uint8)
            #color_img[_i, _j, -1] = (255 * b).astype(np.uint8)

            color_img[_i, _j, :] = (RED * s)
            color_img[_i, _j, -1] = (1-b)

            #color_img[_i+1, _j, :] = (BLUE * d).astype(np.uint8)
            #color_img[_i+1, _j, -1] = (255 * b).astype(np.uint8)

            color_img[_i+1, _j, :] = (BLUE * d)
            color_img[_i+1, _j, -1] = (1-b)

            #color_img[_i, _j+1, :] = (GREEN * ir).astype(np.uint8)
            #color_img[_i, _j+1, -1] = (255 * b).astype(np.uint8)

            color_img[_i, _j+1, :] = (GREEN * ir)
            color_img[_i, _j+1, -1] = (1-b)

            #color_img[_i+1, _j+1, :] = (YELLOW * p).astype(np.uint8)
            #color_img[_i+1, _j+1, -1] = (255 * b).astype(np.uint8)

            color_img[_i+1, _j+1, :] = (YELLOW * p)
            color_img[_i+1, _j+1, -1] = (1-b)


    return color_img

def main():
    sph_m = 'spheroid_mean.fits'
    sph_v = 'spheroid_var.fits'
    dk_m = 'disk_mean.fits'
    dk_v = 'disk_var.fits'
    irr_m = 'irregular_mean.fits'
    irr_v = 'irregular_var.fits'
    ps_m = 'point_source_mean.fits'
    ps_v = 'point_source_var.fits'
    bkg_m = 'background_mean.fits'
    bkg_v = 'background_var.fits'
    n = 'n.fits'


    f_loc = './classify_out/{}'

    bkg = [fits.getdata(f_loc.format(f))[5:-5,5:-5] for f in [bkg_m, bkg_v]]
    sph = [fits.getdata(f_loc.format(f))[5:-5,5:-5] for f in [sph_m, sph_v]]
    dk = [fits.getdata(f_loc.format(f))[5:-5,5:-5] for f in [dk_m, dk_v]]
    irr = [fits.getdata(f_loc.format(f))[5:-5,5:-5] for f in [irr_m, irr_v]]
    ps = [fits.getdata(f_loc.format(f))[5:-5,5:-5] for f in [ps_m, ps_v]]

    for i in range(1,5):
        img = colorize(bkg, sph, dk, irr, ps, color=i, variance_weighting=False)
        plt.figure(figsize=(10,10))
        plt.imshow(img, origin='lower')
        legend_vals = [ patches.Patch(color=RED, label='Spheroid'),
                        patches.Patch(color=BLUE, label='Disk'),
                        patches.Patch(color=GREEN, label='Irregular'),
                        patches.Patch(color=YELLOW, label='Point Source')]
        plt.tick_params(axis='both',
                        bottom='off',
                        labelbottom='off',
                        left='off',
                        labelleft='off')

        plt.legend(handles=legend_vals)
        plt.savefig(f'./figs/{i}-color-classification.pdf', dpi=600)

if __name__=='__main__':
    main()