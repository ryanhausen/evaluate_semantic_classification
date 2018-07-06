import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


prev_m = fits.getdata('./old/background_mean.fits')
prev_v = fits.getdata('./old/background_var.fits')

new_m = fits.getdata('./all_bands/background_mean.fits')
new_v = fits.getdata('./all_bands/background_var.fits')

hj_m = fits.getdata('./hj/background_mean.fits')
hj_v = fits.getdata('./hj/background_var.fits')

for m, v, n in zip([prev_m, new_m, hj_m], [prev_v, new_v, hj_v], ['all 40', 'inner 30', 'inner 30 HJ only']):
    f, (ax1, ax2) = plt.subplots(ncols=2)
    f.suptitle(n)
    ax1.set_title('mean')
    img = ax1.imshow(m, cmap='gray', vmin=0, vmax=1)
    f.colorbar(img, ax=ax1)
    ax2.set_title('variance')
    img = ax2.imshow(v, cmap='gray')
    f.colorbar(img, ax=ax2)
    f.tight_layout()

f, (ax1, ax2) = plt.subplots(ncols=2)
f.suptitle('All 40 - Inner 30')
ax1.set_title('Mean')
img = ax1.imshow(prev_m - new_m, cmap='gray')
f.colorbar(img, ax=ax1)
ax2.set_title('Varaiance')
img = ax2.imshow(prev_v - new_v, cmap='gray')
f.colorbar(img, ax=ax2)
f.tight_layout()

f, (ax1, ax2) = plt.subplots(ncols=2)
f.suptitle('Inner 30 - Inner 30 HJ Only')
ax1.set_title('Mean')
img = ax1.imshow(new_m - hj_m, cmap='gray')
f.colorbar(img, ax=ax1)
ax2.set_title('Varaiance')
img = ax2.imshow(new_v - hj_v, cmap='gray')
f.colorbar(img, ax=ax2)
f.tight_layout()

plt.show()
