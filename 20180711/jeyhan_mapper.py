import os

from astropy.io import fits


def get_label(src):
    with open('labels.csv', 'r') as f:
        for l in f:
            l = l.split(',')
            if l[2]==src:
                return l[3:]

def main():
    large_header = fits.getheader('hlsp_candels_hst_wfc3_gs-tot_f125w_v1.0_drz.fits')
    offset_y, offset_x = 12000, 7300

    src_data = []
    imgs = os.listdir('./jeyhan_imgs')
    for i, img in enumerate(imgs):
        print(i/len(imgs), end='\r')

        header = fits.getheader(f'./jeyhan_imgs/{img}')

        y = int(large_header['CRPIX2'] - header['CRPIX2']) + 42
        x = int(large_header['CRPIX1'] - header['CRPIX1']) + 42

        adj_y, adj_x = y - offset_y, x - offset_x

        src = img.replace('.fits', '')
        lbl = get_label(src)

        str_data = [src, y, x, adj_y, adj_x]
        str_data = [str(s) for s in str_data]
        if lbl:
            str_data.extend(lbl)
        else:
            continue

        src_data.append(','.join(str_data))

    with open('mapped_srcs.csv', 'w')  as f:
        for src in src_data:
            f.write(src)

if __name__=='__main__':
    main()


