import os

import numpy as np
from astropy.io import fits

output = []
for f in os.listdir('./segmaps'):
    a = fits.getdata('./segmaps/{}'.format(f))
    src_id = int(f.replace('.fits', '').split('_')[-1])
    area = (a==src_id).sum()
    r = np.sqrt(area/np.pi)
    output.append('{},{}\n'.format(f.replace('.fits', ''), r))
    
with open('source_radii.csv', 'w') as f:
    f.writelines(output)
