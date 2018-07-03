import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def img_plot(arr, saveto=None):
    img_style = {
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Helvetica'],
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

    cx, cy = int(img_size[0] // 2), int(img_size[1] // 2)
    xs, ys = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))
    rs = np.sqrt((xs-cx)**2 + (ys-cy)**2)
    img_plot(rs, saveto='figs/rs.pdf')

    # make source
    re = 5

    aperature = rs < re
    num_samples = 6

    for factor in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        s = sersic(1, re, rs, 1)

        source = []
        for n in all_noise:
            rms = get_rms(n, aperature, num_samples)
            source.append(s * (factor * rms / s[aperature].sum()))

        fits_write(np.array(source), './fits', f'source-{factor}.fits')

        combined = [all_noise[i]+source[i] for i in range(len(all_noise))]
        fits_write(np.array(combined), './fits', f'combined-{factor}.fits')

if __name__=='__main__':
    main()