import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from skimage.morphology import watershed, local_minima

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

def generate_segmap(bkg):
    # create segmentation maps based on watershedding from 0528
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


def main():

    # get noise
    segmap = fits.getdata('segmap.fits')
    h = fits.getdata('h.fits')
    j = fits.getdata('j.fits')
    v = fits.getdata('v.fits')
    z = fits.getdata('z.fits')

    img_size = [200, 200]
    make_noise = lambda a: np.random.choice(a, size=img_size)

    noise = segmap==1
    all_noise = [make_noise(b[noise]) for b in [h,j,v,z]]
    all_noise = [a + abs(a.min()) + 1e-6 for a in all_noise]

    for b, noise in zip('HJVZ', all_noise):
        img_plot(noise, saveto=f'figs/{b}.pdf')

    re = 5
    src_per_row = img_size[0] // (12*re)
    xs = [x for x in range(6*re, src_per_row*(12*re), 12*re)]

    src_per_col = img_size[1] // (12*re)
    ys = [y for y in range(6*re, src_per_col*(12*re), 12*re)]

    centers = []
    for y in ys:
        for x in xs:
            centers.append((y,x))

    xs, ys = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))

    rs = []
    for cy, cx in centers:
        rs.append(np.sqrt((xs-cx)**2 + (ys-cy)**2))

    aperature_rs = np.sqrt((xs-(img_size[1]//2))**2 + (ys-(img_size[0]//2))**2)
    aperature = aperature_rs < re
    num_samples = 100
    rms_vals = [get_rms(n, aperature, num_samples) for n in all_noise]

    factors = np.linspace(0.5, 10, num=len(rs))
    for i, factor in enumerate(factors):

        s = sersic(1, re, rs[i], 1)

        source = []
        for rms in rms_vals:
            source.append(s * (factor * rms / s[rs[i]<re].sum()))

        all_noise = [all_noise[i]+source[i] for i in range(len(all_noise))]

    fits_write(np.array(all_noise), './fits', 'combined-{}.fits'.format('-'.join([str(n) for n in factors])))

if __name__=='__main__':
    main()

