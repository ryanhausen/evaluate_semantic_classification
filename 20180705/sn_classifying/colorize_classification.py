import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# RED = np.array([255, 0, 0, 0], dtype=np.uint8)
# GREEN = np.array([0, 255, 0, 0], dtype=np.uint8)
# BLUE = np.array([0, 0, 255, 0], dtype=np.uint8)
# YELLOW = np.array([255, 255, 0, 0], dtype=np.uint8)

# RED = np.array([255, 0, 0], dtype=np.uint8)
# GREEN = np.array([0, 255, 0], dtype=np.uint8)
# BLUE = np.array([0, 0, 255], dtype=np.uint8)
# YELLOW = np.array([255, 255, 0], dtype=np.uint8)

RED = np.array([1, 0, 0, 0])
GREEN = np.array([0, 1, 0, 0])
BLUE = np.array([0, 0, 1, 0])
YELLOW = np.array([1, 1, 0, 0])



def colorize(bkg, sph, dk, irr, ps):
    in_y, in_x = bkg.shape
    color_img = np.zeros([in_y*2, in_x*2, 4], dtype=np.float32)


    for i in range(in_y):
        for j in range(in_x):
            b = bkg_m[i,j]
            s = sph_m[i,j]
            d = dk_m[i,j]
            ir = irr_m[i,j]
            p = ps_m[i,j]

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
    img = colorize(fits.getdata(f_loc.format(bkg_m)),
                   fits.getdata(f_loc.format(sph_m)),
                   fits.getdata(f_loc.format(dk_m)),
                   fits.getdata(f_loc.format(irr_m)),
                   fits.getdata(f_loc.format(ps_m)))

    plt.imshow(img, origin='lower')
    plt.show()

if __name__=='__main__':
    main()