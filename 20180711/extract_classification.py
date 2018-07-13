import os
import shutil

import numpy as np
from astropy.io import fits
from skimage.morphology import watershed

def find_local_minima(a):
    _a = np.zeros_like(a, dtype=np.int)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            n = get_neighborhood(a, i, j)
            if n.min()==a[i,j]:
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



def morphological_classfication(morph_means, morph_var, flux, src_map):
    return scheme_flux_weighted_simple_mean(morph_means, morph_var, src_map)

def scheme_pixel_rank(morph_means, morph_var, src_map, rank):

    not_bkg = 1 - morph_means[-1, :, :]

    weighted_vals = morph_means[:-1,:,:] / morph_var[:-1,:,:]
    weighted_vals /= weighted_vals.sum(axis=0)

    sorted_vals = np.argsort(-weighted_vals, axis=0)

    classification = np.zeros([4])
    for r in range(rank):
        for i in range(sorted_vals.shape[1]):
            for j in range(sorted_vals.shape[2]):
                if src_map[i,j]:
                    morph = sorted_vals[r,i,j]
                    classification[morph] += not_bkg[i,j] * weighted_vals[morph,i,j]

    return classification / classification.sum()

def scheme_flux_weighted_simple_sum(morph_means, morph_var, flux, src_map):
    num_classes = 4
    classification = np.zeros([num_classes])
    for i in range(num_classes):
        classification[i] = (morph_means[i,:,:] * flux * src_map).sum()

    return classification / classification.sum()

def scheme_flux_weighted_simple_mean(morph_means, morph_var, flux, src_map):
    num_classes = 4
    classification = np.zeros([num_classes])
    for i in range(num_classes):
        classification[i] = (morph_means[i,:,:] * flux * src_map).mean()

    return classification / classification.sum()

def scheme_simple_mean(morph_means, morph_var, src_map):
    num_classes = 4
    classification = np.zeros([num_classes])
    not_bkg = 1 - morph_means[-1, :, :]
    morph_var[morph_var==0] = 1
    for i in range(num_classes):
        classification[i] = (morph_means[i,:,:] * src_map).mean()

    return classification / classification.sum()

def scheme_doubly_variance_weighted_mean(morph_means, morph_var, src_map):
    num_classes = 4
    classification = np.zeros([num_classes])
    class_variance = np.zeros([num_classes])
    not_bkg = 1 - morph_means[-1, :, :]
    morph_var[morph_var==0] = 1
    for i in range(num_classes):
        numerator = not_bkg * morph_means[i,:,:] / morph_var[i,:,:]
        denominator =  not_bkg / morph_var[i,:,:]
        numerator = (numerator * src_map).sum()
        denominator = (denominator * src_map).sum()

        classification[i] = numerator/denominator
        class_variance[i] = denominator

    classification = classification / class_variance

    return classification / classification.sum()

def scheme_variance_weighted_mean_flux_weighted(morph_means, morph_var, flux, src_map):
    num_classes = 4
    classification = np.zeros([num_classes])
    not_bkg = 1 - morph_means[-1, :, :]
    morph_var[morph_var==0] = 1
    flux_map = flux * src_map
    for i in range(num_classes):
        numerator = not_bkg * morph_means[i,:,:] / morph_var[i,:,:]
        denominator =  not_bkg / morph_var[i,:,:]
        numerator = (flux_map * numerator * src_map).sum()
        denominator = (denominator * src_map).sum()
        classification[i] = numerator/denominator

    return classification / classification.sum()

def scheme_variance_weighted_mean(morph_means, morph_var, src_map):
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
        var_vals.append(data[f'{m}_var'][sy, sx].copy())

    return np.array(mean_vals), np.array(var_vals), data['flux'][sy, sx].copy()

def main():
    size = (84,84)

    hduls = []
    data = {}
    for f in os.listdir('./classifications'):
        hdul = fits.open(f'./classifications/{f}', memmap=True, mode='readonly')
        data[f.replace('.fits', '')] = hdul[0].data

    with fits.open('./input/small125.fits', memmap=True, mode='readonly') as hdul:
        data['flux'] = hdul[0].data

    with open('mapped_srcs.csv', 'r') as f, open('output_summary.csv', 'w') as g:
        lines = f.readlines()
        total = len(lines)
        for i, src_data in enumerate(lines):
            print(i/total, end='\r')
            src_data = src_data.strip().split(',')
            src_name = src_data[0]
            y, x = int(src_data[3]), int(src_data[4])
            mean, var, flux = make_slices(data, y, x, size)

            segmap, src_id = create_segmap(mean[-1,:,:])

            classification = morphological_classfication(mean, var, flux, segmap==src_id)
            src_data += [str(c) for c in classification]
            g.write(','.join(src_data) + '\n')

            new_path = f'./output/{src_name}'
            os.mkdir(new_path)

            f_names = [f'segmap-{src_id}.fits', 'means.fits', 'vars.fits']
            f_data = [segmap, mean, var]
            for n, d in zip(f_names, f_data):
                fits.PrimaryHDU(d).writeto(os.path.join(new_path, n))

            shutil.copyfile(f'./jeyhan_imgs/{src_name}.fits',
                            os.path.join(new_path, f'{src_name}.fits'))


    for hdul in hduls:
        hdul.close()

if __name__=='__main__':
    main()