import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

with open('valid_slices', 'r') as f:
    valid_slices = [int(r) for r in f.readlines()]

probs = fits.getdata('slices_labels.fits')
imgs = fits.getdata('slices.fits')

for s in valid_slices:
    f, a = plt.subplots(nrows=2, ncols=4)
    a = np.array(a).flatten()
    
    for i in range(6):
        a[i].imshow(probs[s,:,:,i], cmap='gray', origin='lower', vmax=1, vmin=0)
    a[6].imshow(imgs[s,:,:,0], cmap='gray', origin='lower')
        
    plt.show()
