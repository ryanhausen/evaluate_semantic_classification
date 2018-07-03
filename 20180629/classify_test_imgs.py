import os

import numpy as np
from astropy.io import fits
from classify import classify_img


def move_output(dest):
    out_files = [
        'spheroid_mean.fits',
        'spheroid_var.fits',
        'disk_mean.fits',
        'disk_var.fits',
        'irregular_mean.fits',
        'irregular_var.fits',
        'point_source_mean.fits',
        'point_source_var.fits',
        'background_mean.fits',
        'background_var.fits',
        'n.fits'
    ]

    for out_file in out_files:
        dest_file = os.path.join(dest, out_file)
        os.rename(out_file, dest_file)

def move_to_input(test_img):
    f_dir = './classify_in/{}.fits'
    for i, b in enumerate('hjvz'):
        fits.PrimaryHDU(test_img[i,:,:]).writeto(f_dir.format(b))

def main():
    # get img
    img_dir = './fits'
    for test_file in os.listdir('./fits'):
        print(f'Classifying {test_file}...')
        with fits.open(os.path.join(img_dir, test_file)) as hdul:
            move_to_input(hdul[0].data)
        del hdul[0].data

        h = './classify_in/h.fits'
        j = './classify_in/j.fits'
        v = './classify_in/v.fits'
        z = './classify_in/z.fits'

        classify_img(h=h, j=j, v=v, z=z)

        out_dir = './classify_out/{}'.format(test_file.replace('.fits',''))
        os.mkdir(out_dir)
        move_output(out_dir)

        for i in [h, j, v, z]:
            os.remove(i)



if __name__=='__main__':
    main()