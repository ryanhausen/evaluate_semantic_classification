import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.morphology import watershed, local_minima, local_maxima, h_minima

def img_plot(arr, saveto=None, cmap='gray', colorbar=True):
    img_style = {
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Helvetica-normal'],
        'image.cmap':       cmap,
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
        if colorbar:
            f.colorbar(img, ticks=[arr.min(), arr.max()])

        if saveto:
            plt.savefig(saveto)

def make_segmap(bkg):
    # create segmentation maps based on watershedding from 0528
    local_min = find_local_minima(bkg)
    img_plot(local_min)
    #local_max = local_maxima(bkg)
    mask = bkg<0.3
    masked_min = mask * local_min
    img_plot(masked_min)

    count = 0
    for i in range(bkg.shape[0]):
        for j in range(bkg.shape[1]):
            if masked_min[i,j] == 1:
                count += 1
                masked_min[i,j] = count

    masked_min = masked_min + local_min
    segmap = watershed(bkg, masked_min, compactness=0.01)

    return segmap

def find_local_minima(a):
    _a = np.zeros_like(a, dtype=np.int)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            n = get_neighborhood(a, i, j)
            mid = n.shape[0]//2
            if n.min()==n[mid,mid]:
                _a[i,j] = True

    return _a


def get_neighborhood(a, i, j):
    size = 3

    y_min = max(i-size, 0)
    y_max = min(i+size+1, a.shape[0])

    x_min = max(j-size, 0)
    x_max = min(j+size+1, a.shape[1])

    neighborhood =  a[y_min:y_max, x_min:x_max]

    return neighborhood

def morphological_classfication(morph_means, morph_var, src_map):
    num_classes = 4
    classification = np.zeros([num_classes])
    not_bkg = 1 - morph_means[-1, :, :]
    morph_var[morph_var==0] = 1
    for i in range(num_classes):
        numerator = not_bkg * morph_means[i,:,:] / morph_var[i,:,:]
        denominator =  not_bkg / morph_var[i,:,:]
        numerator = (numerator * src_map).sum()
        denominator = (denominator * src_map).sum()
        classification[i] = numerator/denominator
    # for i in range(num_classes):
    #     classification[i] = (morph_means[i,:,:] * src_map).sum()



    return classification / classification.sum()

def get_means_vars(path):
    morphs = ['spheroid', 'disk', 'irregular', 'point_source']

    mean = [fits.getdata(os.path.join(path, f'{m}_mean.fits'))[5:-5, 5:-5] for m in morphs]
    var = [fits.getdata(os.path.join(path, f'{m}_var.fits'))[5:-5, 5:-5] for m in morphs]

    return np.array(mean), np.array(var)

def main():
    bkg = fits.getdata('./classify_out/background_mean.fits')[5:-5, 5:-5]
    mean, var = get_means_vars('./classify_out')

    segmap = make_segmap(bkg)
    img_plot(1-bkg, saveto='./figs/background.pdf')
    img_plot(segmap, saveto='./figs/segmap.pdf', cmap='cubehelix', colorbar=False)



    sn = [0.5, 1.68, 2.875, 4.06, 5.25, 6.43, 7.63, 8.81, 10]
    sph, disk, irr, ps = [], [], [], []
    for src in range(2, segmap.max()+1):
        classifications = morphological_classfication(mean, var, segmap==src)
        for c, cl in zip(classifications, [sph, disk, irr, ps]):
            cl.append(c)

    plt.figure()
    plt.plot(sn, sph, color='r', label='Spheroid')
    plt.plot(sn, disk, color='b', label='Disk')
    plt.plot(sn, irr, color='g', label='Irregular')
    plt.plot(sn, ps, color='y', label='Point Source')
    plt.xlabel('Sersic Index (n)')
    plt.ylabel('P$(src=class)$')
    plt.legend()
    plt.savefig('./figs/morph.pdf')




if __name__=='__main__':
    main()