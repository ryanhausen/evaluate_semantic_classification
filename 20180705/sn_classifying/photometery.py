
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.morphology import watershed, local_minima, local_maxima, h_minima

def make_segmap(bkg):
    # create segmentation maps based on watershedding from 0528
    s = np.zeros([5,5])
    s[2,2] = 1
    local_min = local_minima(bkg)
    plt.imshow(local_min)
    #local_max = local_maxima(bkg)
    mask = bkg<0.3
    masked_min = mask * local_min
    plt.figure()
    plt.imshow(masked_min)
    count = 0
    for i in range(bkg.shape[0]):
        for j in range(bkg.shape[1]):
            if masked_min[i,j] == 1:
                v = check_neighbors(masked_min, i, j)
                if v > 1:
                    masked_min[i,j] = v
                else:
                    count += 1
                    masked_min[i,j] = count

    masked_min = masked_min + local_min
    plt.figure()
    plt.imshow(masked_min)
    segmap = watershed(bkg, masked_min)

    return segmap

def check_neighbors(a, i, j):

    neighborhood =  a[max(i-1, 0):min(a.shape[0], i+1), max(0, j-1):min(a.shape[1], j+1)]
    if (neighborhood > 1).sum() > 0:
        return neighborhood[neighborhood>1][0]
    else:
        return 1

def main():
    bkg = fits.getdata('./classify_out/background_mean.fits')
    segmap = make_segmap(bkg[5:-5, 5:-5])

    plt.figure()
    plt.imshow(segmap, cmap='gray', origin='lower')
    plt.show()



if __name__=='__main__':
    main()