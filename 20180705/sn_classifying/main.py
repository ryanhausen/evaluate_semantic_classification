import os

from PIL import Image
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

import generate_source

def img_plot(arr, saveto=None):
    img_style = {
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Helvetica-normal'],
        'image.cmap':       'gray',
        'image.origin':     'lower',
        'savefig.dpi':      600,
        'savefig.format':   'pdf',
    }
    with plt.style.context(img_style):
        f  = plt.figure()

        plt.tick_params(axis='both',
                        bottom='off',
                        labelbottom='off',
                        left='off',
                        labelleft='off')
        img = plt.imshow(arr)
        f.colorbar(img, ticks=[arr.min(), arr.max()])

        if saveto:
            plt.savefig(saveto)

def get_rms(noise_img, aperature, num_samples):
    ax0_buffer = np.sum(aperature, axis=0).max() / 2
    ax1_buffer = np.sum(aperature, axis=1).max() / 2
    ax0_shift = int(noise_img.shape[0]//2 - ax0_buffer)
    ax1_shift = int(noise_img.shape[1]//2 - ax1_buffer)
    rms_vals = []
    for _ in range(num_samples):
        s0 = np.random.randint(ax0_shift) * 1 if np.random.randint(2) else -1
        s1 = np.random.randint(ax1_shift) * 1 if np.random.randint(2) else -1

        shifted_aperature = np.roll(aperature, (s0,s1), axis=(0,1))

        rms_vals.append(noise_img[shifted_aperature].sum())

    return np.sqrt(np.mean(np.square(rms_vals)))

def sersic(Ie, Re, R, m):
    bm = 2.0*m - 0.324
    return Ie * np.exp(-bm * ((R/Re)**(1/m) - 1))

def fits_write(a, path, file_name):
    if file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))
    fits.PrimaryHDU(a).writeto(os.path.join(path, file_name))

def _to_origin(y, x):
    return np.array([
                [1.0, 0.0, x],
                [0.0, 1.0, y],
                [0.0, 0.0, 1.0]
            ])

def _from_origin(y, x):
    return np.array([
                [1.0, 0.0, -x],
                [0.0, 1.0, -y],
                [0.0, 0.0, 1.0]
            ])

def _scale_image(w, h):
    return np.array([
                [w, 0.0, 0.0],
                [0.0, h, 0.0],
                [0.0, 0.0, 1.0]
            ])

def PIL_tuple(matrix):
    return tuple(matrix.flatten()[:6])

def elliptical_transform(img, center, re, ratio):
    w = ratio * 4
    h = w/ratio
    print(w, h)

    a = _to_origin(*center)
    b = _scale_image(h , w)
    c = _from_origin(*center)

    im = Image.fromarray(img)
    im = im.transform(img.shape, Image.AFFINE, data=PIL_tuple(a.dot(b).dot(c)))
    return np.array(im)

def main():
    # get noise
    segmap = fits.getdata('segmap.fits')
    h = fits.getdata('h.fits')
    j = fits.getdata('j.fits')
    v = fits.getdata('v.fits')
    z = fits.getdata('z.fits')

    tt_h = fits.getdata('tinytim/h.fits')
    tt_j = fits.getdata('tinytim/j.fits')
    tt_v = fits.getdata('tinytim/v.fits')
    tt_z = fits.getdata('tinytim/z.fits')
    tiny_tims = [tt_h, tt_j, tt_v, tt_z]

    img_size = [250, 250]
    make_noise = lambda a: np.random.choice(a, size=img_size)

    noise = segmap==1
    all_noise = [make_noise(b[noise]) for b in [h,j,v,z]]
    all_noise = [a + abs(a.min()) + 1e-6 for a in all_noise]

    for b, noise in zip('HJVZ', all_noise):
        img_plot(noise, saveto=f'figs/{b}.pdf')

    re = 5
    src_per_row = img_size[0] // (12*re)
    src_xs = [x for x in range(12*re, src_per_row*(12*re), 12*re)]

    src_per_col = img_size[1] // (12*re)
    src_ys = [y for y in range(12*re, src_per_col*(12*re), 12*re)]
    xs, ys = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))


    centers = []
    rs = []
    for y in src_ys:
        for x in src_xs:
            centers.append((y,x))
            rs.append(np.sqrt((xs-x)**2 + (ys-y)**2))

    aperature_rs = np.sqrt((xs-(img_size[1]//2))**2 + (ys-(img_size[0]//2))**2)
    aperature = aperature_rs < re
    num_samples = 1000
    rms_vals = [get_rms(n, aperature, num_samples) for n in all_noise]

    factors = np.linspace(1, 10, num=len(centers))
    for i, factor in enumerate(factors):

        s = generate_source.exponential(img_size,
                                        centers[i][1],
                                        centers[i][0],
                                        1,
                                        re,
                                        simple=False)

        source = []
        for j, rms in enumerate(rms_vals):
            #s = convolve(s, tiny_tims[j])
            src_adj = s * (factor * rms / s[rs[i]<re].sum())
            src_adj = convolve(src_adj, tiny_tims[j])
            #src_adj = elliptical_transform(src_adj, centers[i], re, factor)
            source.append(src_adj)

        all_noise = [all_noise[i]+source[i] for i in range(len(all_noise))]

    fits_write(np.array(all_noise), './fits', 'combined-{}.fits'.format('-'.join([str(n) for n in factors])))

if __name__=='__main__':
    main()

