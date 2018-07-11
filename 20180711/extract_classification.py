import os

import numpy as np
from astropy.io import fits
from skimage.morphology import watershed

def find_local_minima(a):
    _a = np.zeros_like(a, dtype=np.int)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            n = get_neighborhood(a, i, j)
            mid = n.shape[0]//2
            if n.min()==n[mid,mid]:
                _a[i,j] = True

    return _a


def create_segmap(bkg):
    # create segmentation maps based on watershedding from 0528
    local_min = find_local_minima(bkg)

    #local_max = local_maxima(bkg)
    mask = bkg<0.3
    masked_min = mask * local_min

    count = 0
    for i in range(bkg.shape[0]):
        for j in range(bkg.shape[1]):
            if masked_min[i,j] == 1:
                count += 1
                masked_min[i,j] = count

    masked_min = masked_min + local_min
    segmap = watershed(bkg, masked_min, compactness=0.01)

    return segmap, segmap[bkg.shape[0]//2, bkg.shape[1]//2]

def get_neighborhood(a, i, j):
    size = 3

    y_min = max(i-size, 0)
    y_max = min(i+size+1, a.shape[0])

    x_min = max(j-size, 0)
    x_max = min(j+size+1, a.shape[1])

    return a[y_min:y_max, x_min:x_max]



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

    return classification / classification.sum()

def make_slices(data, y, x, size):
    morphs = ['spheroid', 'disk', 'irregular', 'point_source', 'background']

    buffer_y = size[0]//2
    buffer_x = size[1]//2

    sy, sx = slice(y-buffer_y, y+buffer_y), slice(x-buffer_x, x+buffer_x)


    mean_vals = []
    var_vals = []

    for m in morphs:
        mean_vals.append(data[f'{m}_mean'][sy, sx].copy())
        var_vals.append(data[f'{m}_mean'][sy, sx].copy())

    return np.array(mean_vals), np.array(var_vals)

def main():
    size = (84,84)

    hduls = []
    data = {}
    for f in os.listdir('./classifications'):
        hdul = fits.open(f'./classifications/{f}', memmap=True, mode='readonly')
        data[f.replace('.fits', '')] = hdul[0].data

    with open('mapped_srcs.csv', 'r') as f:
        for src_data in f:
            src_data = src_data.split(',')
            src_name = src_data[0]
            y, x = int(src_data[3]), int(src_data[4])
            mean, var = make_slices(data, y, x, size)

            segmap, src_id = create_segmap(mean[-1,:,:])









if __name__=='__main__':
    main()