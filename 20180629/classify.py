import argparse
import os
import time
from collections import namedtuple

import numpy as np
from astropy.io import fits
import tensorflow as tf

from convnet_semantic_segmentation import Model
import tf_logger as log
tf.logging.set_verbosity(tf.logging.INFO)

# Mean and Variance calculations from:
# http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.8508&rep=rep1&type=pdf

# Working with large fits files
# http://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html

INPUT_W, INPUT_H = 40, 40
# TF vars
x = tf.placeholder(tf.float32, shape=[None,40,40,1])
global classifier
classifier = None
global tf_session
tf_session = None

def get_pretty_time(total_time, count, num_batches):
    avg_time_per_batch = total_time/count
    est_sec = avg_time_per_batch * (num_batches-count)
    est_min = est_sec // 60
    est_hrs = est_min // 60
    est_dys = est_hrs // 24

    time_string = '{} days, {} hours, {} minutes, {} seconds'
    return time_string.format(est_dys, est_hrs, est_min, round(est_sec%60, 2))


def classify_img(*ignore,
                 h='h.fits',
                 j='j.fits',
                 v='v.fits',
                 z='z.fits',
                 batch_size=1000):
    naxis1, naxis2 = _validate_args(ignore, h, j, v, z)
    hduls, data_handles = _prepare_files(naxis1, naxis2, h, j, v, z)

    bands = data_handles[:4]
    n = data_handles[4]
    # sph, dk, irr, ps, bkg
    morphologies = [(data_handles[i], data_handles[i+1]) for i in range(5,len(data_handles), 2)]

    final_y = naxis2-INPUT_H
    final_x = naxis1-INPUT_W

    index_gen = _index_generator(final_y+1, final_x+1)

    num_batches = final_x*final_y//batch_size
    count = 1
    print('Total number of batches to be processed {}'.format(num_batches))
    progress = '{}% est time: {}'

    total_time = 0
    while True:
        complete = round(count/num_batches, 4)
        start = time.time()
        print(progress.format(complete, get_pretty_time(total_time,count,num_batches)), end='\r')
        count += 1


        batch = []
        batch_idx = []

        done = False
        for _ in range(batch_size):
            try:
                y, x = next(index_gen)
            except StopIteration:
                print('\nImage Done')
                done = True
                break
            combined = [b[y:y+INPUT_H,x:x+INPUT_W] for b in bands]
            batch.append(_pre_process(np.array(combined)))
            batch_idx.append((y,x))

        if len(batch)==0:
            break

        batch = np.array(batch)

        # classify batch
        labels = _run_classifier(batch)

        for i, l in enumerate(labels):
            y, x = batch_idx[i]
            ns = n[y:y+INPUT_H, x:x+INPUT_W] + 1
            n[y:y+INPUT_H, x:x+INPUT_W] = ns
            final_map = _get_final_map(naxis2, naxis1, y, x)

            for j, m in enumerate(morphologies):
                x_n = l[:,:,j]
                prev_mean = m[0][y:y+INPUT_H, x:x+INPUT_W]
                prev_var = m[1][y:y+INPUT_H, x:x+INPUT_W]

                curr_mean, curr_var = _get_updates(ns,
                                                    x_n,
                                                    prev_var,
                                                    prev_mean,
                                                    final_map)

                m[0][y:y+INPUT_H, x:x+INPUT_W] = curr_mean
                m[1][y:y+INPUT_H, x:x+INPUT_W] = curr_var

        total_time += time.time()-start
        if done:
            break

    for hdul in hduls:
        hdul.close()

def _index_generator(upto_y, upto_x):
    for y in range(upto_y):
        for x in range(upto_x):
            yield (y, x)


# store classifier as a singleton?
def _run_classifier(batch):
    """
    Takes a batch of images returns the classifications of the same images
    INPUT:
    batch: numpy array [n,40,40,1]

    RETURNS
    classified images: numpy array [n,40,40,5]
    """
    global classifier
    global tf_session
    if classifier is None:
        # mock a dataset, maybe move this to a tf dataset?
        DataSet = namedtuple('Dataset', ['NUM_LABELS'])
        d = DataSet(5)

        Model.DATA_FORMAT = 'channels_last'
        classifier = Model(d, False).inference(x)
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf_session = tf.Session(config=config)
        saver.restore(tf_session, tf.train.latest_checkpoint('./weights'))

    return tf_session.run(classifier, feed_dict={x:batch})


def _validate_out_files = [
        'spheroid_mean.fits'
        'spheroid_var.fits'
        'disk_mean.fits'
        'disk_var.fits'
        'irregular_mean.fits'
        'irregular_var.fits'
        'point_source_mean.fits'
        'point_source_var.fits'
        'background_mean.fits'
        'background_var.fits'
        'n.fits'
    ]
    if ignore:out_files = [
        'spheroid_mean.fits'
        'spheroid_var.fits'
        'disk_mean.fits'
        'disk_var.fits'
        'irregular_mean.fits'
        'irregular_var.fits'
        'point_source_mean.fits'
        'point_source_var.fits'
        'background_mean.fits'
        'background_var.fits'
        'n.fits'
    ]
        raise Exception('Positional arguments are not allowed, please use named arguments')

    # validate files names
    for fname in [h,j,v,z]:
        if not os.path.exists(fname) or not os.path.isfile(fname):
            raise ValueError('Invalid file {}'.format(fname))

    # validate file sizes
    naxis1, naxis2 = None, None
    for fname in [h,j,v,z]:
        hdul = fits.open(fname)
        hdu = hdul[0]
        if naxis1 is None:
            naxis1 = hdu.header['NAXIS1']
            naxis2 = hdu.header['NAXIS2']
        else:
            if naxis1 != hdu.header['NAXIS1']:
                msg = 'Images are not same dims NAXIS1: {} != {}'
                msg = msg.format(naxis1, hdu.header['NAXIS1'])
                raise Exception(msg)
            elif naxis2 != hdu.header['NAXIS2']:
                msg = 'Images are not same dims NAXIS2: {} != {}'
                msg = msg.format(naxis1, hdu.header['NAXIS2'])
                raise Exception(msg)

        hdul.close()

    return naxis1, naxis2

def _prepare_files(naxis1, naxis2, h, j, v, z):
    # create fits files to fill with values
    sph_m = 'spheroid_mean.fits'
    sph_v = 'spheroid_var.fits'
    dk_m = 'disk_mean.fits'
    dk_v = 'disk_var.fits'
    irr_m = 'irregular_mean.fits'
    irr_v = 'irregular_var.fits'
    ps_m = 'point_source_mean.fits'
    ps_v = 'point_source_var.fits'
    bkg_m = 'background_mean.fits'
    bkg_v = 'background_var.fits'
    n = 'n.fits'

    f_names = [sph_m, sph_v, dk_m, dk_v, irr_m, irr_v, ps_m, ps_v, bkg_m, bkg_v]
    for f in f_names:
        _create_file(f, naxis1, naxis2, np.float32)

    _create_file(n, naxis1, naxis2, np.int16)

    hduls = []
    data_handles = []

    for f in [h,j,v,z,n] + f_names:
        print('Reading {}...'.format(f))
        hdul = fits.open(f,
                         memmap=True,
                         mode='readonly' if f in [h,j,v,z] else 'update')
        hduls.append(hdul)
        data_handles.append(hdul[0].data)

    return hduls, data_handles

def _get_updates(n, x_n, prev_var, prev_mean, final_map):
    curr_mean = _iterative_mean(n, prev_mean, x_n)

    curr_var = _iterative_variance(n,
                                   prev_var,
                                   x_n,
                                   prev_mean,
                                   curr_mean)

    curr_var = _finalize_variance(n, curr_var, final_map)

    return curr_mean, curr_var

def _get_final_map(naxis1, naxis2, y, x):
    final_map = [(0,0)]
    # final_map = [(y,x)]
    end_x = x==(naxis2 - INPUT_W)
    end_y = y==(naxis1 - INPUT_H)

    if end_x:
        # final_map.extend([(y,_x) for _x in range(x+1, naxis2)])
        final_map.extend([(0,_x) for _x in range(1, 40)])
    if end_y:
        final_map.extend([(_y,0) for _y in range(1, 40)])
    if end_x and end_y:
        for _x in range(1, 40):
            for _y in range(1, 40):
                final_map.append((_y, _x))

    return final_map


# http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, eq. 4
def _iterative_mean(n, prev_mean, x_n):
    return prev_mean + (x_n - prev_mean)/n


# http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, eq. 24
# the final variance calculation needs to be divided by n
def _iterative_variance(n, prev_var, x_n, prev_mean, curr_mean):
    return prev_var + (x_n - prev_mean) * (x_n - curr_mean)

def _finalize_variance(n, curr_var, final_map):
    final_n = np.ones_like(n)
    for y,x in final_map:
        final_n[y,x] = n[y,x]

    return curr_var / final_n


def _pre_process(img):
    img = (img - img.mean()) / max(img.std(), 1/np.sqrt(np.prod(img.shape)))
    return np.mean(img, axis=0)[:,:,np.newaxis]

# http://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html
def _create_file(f_name, naxis1, naxis2, dtype):
    print('Making {}...'.format(f_name))
    stub = np.zeros([100,100], dtype=dtype)

    hdu = fits.PrimaryHDU(data=stub)
    header = hdu.header
    while len(header) < (36 * 4 - 1):
        header.append()
    header['NAXIS1'] = naxis1
    header['NAXIS2'] = naxis2
    header.tofile(f_name)

    bytes_per_value = 0

    if dtype==np.uint8:
        bytes_per_value = 1
    elif dtype==np.int16:
        bytes_per_value = 2
    elif dtype==np.float32:
        bytes_per_value = 4
    elif dtype==np.float64:
        bytes_per_value = 8

    if bytes_per_value==0:
        raise Exception('Didn\'t assign bytes_per_value')

    with open(f_name, 'rb+') as f:
        f.seek(len(header.tostring()) + (naxis1 * naxis2 * bytes_per_value) - 1)
        f.write(b'\0')



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h', help='fits file containing img in H band')
    parser.add_argument('j', help='fits file containing img in J band')
    parser.add_argument('v', help='fits file containing img in V band')
    parser.add_argument('z', help='fits file containing img in Z band')
    parser.add_argument('--batch_size',
                        help='classifier batch size',
                        type=int,
                        default=100)

    args = parser.parse_args()
    try:
        start = time.time()
        classify_img(h=args.h,
                     j=args.j,
                     v=args.v,
                     z=args.z,
                     batch_size=args.batch_size)
        done = time.time() - start
        print('\nCompleted in {} seconds'.format(done))
    except Exception as e:
        print(e)
        print('Cleaning up')
        for f in os.listdir():
            if f.endswith('.fits'):
                os.remove(f)
        raise e
