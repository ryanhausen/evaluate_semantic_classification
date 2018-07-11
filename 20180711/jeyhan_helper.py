#https://www.swift.psu.edu/secure/toop/convert.php
# used to convert from ds9 to decimal values
import shutil

from astropy.io import fits

offset_y = 12000
offset_x = 7300

# table2.data useful columns
col_depth = 0 # 2, 4, 10
col_area = 1 # deep, wide, ers
col_obj_name = 2 # GDS_
col_ra = 3 # Right Ascension
col_dec = 4 # Decliniation
col_obj_id = 5 # Sequential Objec Identifier
col_hmag = 6 # H band magnitude

ra_range = 52.9875, 53.2
dec_range = -27.9422, -27.6917
in_range = lambda r, d: r > ra_range[0] and r < ra_range[1] and d > dec_range[0] and d < dec_range[1]

included_srcs = []
with open('table2.dat') as f:
    f = f.readlines()
    total = len(f)
    for i, line in enumerate(f):
        print('Processing table2.dat {}.....'.format(i/total), end='\r')
        line = line.strip()
        cols = line.split()
        ra = float(cols[col_ra])
        dec = float(cols[col_dec])

        if in_range(ra, dec):
            included_srcs.append(cols[col_obj_name])

orig_f = '/home/ryanhausen/Documents/astro_data/orig_images/{}_h.fits'
dest_f = './jeyhan_imgs/{}.fits'
errors = []
print()
total = len(included_srcs)
for i, src in enumerate(included_srcs):
    print('Getting Sources {}.....'.format(i/total), end='\r')
    try:
        shutil.copyfile(orig_f.format(src), dest_f.format(src))
    except Exception as e:
        errors.append(str(e))

print('\nThe following sources had errors:')
for e in errors:
    print(e)