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
print(np.sum(all_noise[0]<0))
all_noise = [a + abs(a.min()) + 1e-6 for a in all_noise]

for b, noise in zip('HJVZ', all_noise):
    img_plot(noise, saveto=f'{b}.pdf')

if 'H_Noise.fits' in os.listdir():
    all_noise[0] = fits.getdata('H_Noise.fits')
else:
    fits.PrimaryHDU().writeto('H_Noise.fits')

cx, cy = int(img_size[0] // 2), int(img_size[1] // 2)
xs, ys = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))
rs = np.sqrt((xs-cx)**2 + (ys-cy)**2)
img_plot(rs, saveto='rs.pdf')

# make source

re = 5

aperature = rs < re
num_samples = 6
ax0_shift = int(img_size[0]//2 - re)
ax1_shift = int(img_size[1]//2 - re)
rms_vals = []
for _ in range(num_samples):
    s0 = np.random.randint(ax0_shift) * 1 if np.random.randint(2) else -1
    s1 = np.random.randint(ax1_shift) * 1 if np.random.randint(2) else -1

    shifted_aperature = np.roll(aperature, (s0,s1), axis=(0,1))
    img_plot(shifted_aperature)

    rms_vals.append(all_noise[0][shifted_aperature].sum())

rms = np.sqrt(np.mean(np.square(rms_vals)))
print(rms, rms_vals)

def sersic(Ie, Re, R, m):
    bm = 2.0*m - 0.324
    return Ie * np.exp(-bm * ((R/Re)**(1/m) - 1))

factor = 1
s = sersic(1, re, rs, 1)
s *= factor * rms / (s[aperature].sum())

fits.PrimaryHDU(s).writeto('source-{}.fits.'.format(factor))

plt.figure()
plt.hist(all_noise[0].flatten(), bins=100, alpha=0.7, label='Noise')
plt.hist(s[aperature].flatten(), alpha=0.7, label='source')
plt.yscale('log', nonposy='clip')
plt.legend()
plt.savefig('noise_source-{}.png'.format(factor))

fits.PrimaryHDU(s + all_noise[0]).writeto('combined-{}.fits'.format(factor))


print(s[aperature].sum()/rms)

img_plot(s, saveto='source.pdf')

img_plot(s + all_noise[0], saveto='h_combined.pdf')

# save files to be classified