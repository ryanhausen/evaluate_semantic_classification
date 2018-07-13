import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.morphology import watershed, local_minima


bkg = fits.getdata('background_mean.fits')

local_min = local_minima(bkg)
mask = bkg<0.3
masked_min = mask * local_min
count = 0
for i in range(bkg.shape[0]):
    for j in range(bkg.shape[1]):
        if masked_min[i,j] == 1:
            count += 1
            masked_min[i,j] = count
            
masked_min = masked_min + local_min
segmap = watershed(bkg, masked_min)

data = {}
min_src, max_src = 2, segmap.max()+1

for i in range(min_src, max_src):
    data[i] = {}

    y, x = np.where(masked_min==i)
    y, x = y[0], x[0]
    data[i]['c'] = (y,x)
    for c in 'rphjvz':
        data[i][c] = []

    for j in range(bkg.shape[1]):
        if segmap[y, j] == i:
            data[i]['r'].append(x-j)
            data[i]['p'].append(1-bkg[y,j])



    
#dist = lambda p1,p2: ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
#dist = lambda p1,p2: (p1[1]-p2[1])



#for y in range(bkg.shape[0]):
#    for x in range(bkg.shape[1]):
#        src = segmap[y,x]
#        if src in data.keys():
#            data[src]['r'].append(dist(data[src]['c'], (y,x)))
#            data[src]['p'].append(1-bkg[y,x])
            
            
plt.figure()
plt.title('Pixel Probability Profile')
plt.xlabel('Distance from center along X axis (Pixels)')
plt.ylabel('(1-bkg)')
for i in range(min_src, max_src):
    src = data[i]
    args = np.argsort(src['r'])
    xs = sorted(src['r'])
    ys = np.array(src['p'])[args]
    
    
    plt.plot(xs, ys, alpha=0.5)
    #plt.plot(xs, np.poly1d(np.polyfit(xs,ys,2))(xs), alpha=0.5)

plt.show()
    
            
