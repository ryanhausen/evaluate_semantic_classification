import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_offset = 172
y_offset = 269

vals = pd.read_csv('points.csv')

morphologies = ['Background','Disk','Irregular',
                'Point Source','Spheroid','Unknown']




for i in range(84):
    for j in range(84):
        mask = (vals['x'].values-x_offset==i) & (vals['y'].values-y_offset==j)

        pixel_probs = vals.loc[mask, morphologies]
        
        count = sum(mask)
        if count==100:
            print('Count for ({},{})={}'.format(j,i,count), end='\r')
        else:
            print('Count for ({},{})={}'.format(j,i,count))
            
        pixel_probs.to_csv('./separate_probs/{}-{}.tsv'.format(j,i), sep='\t')
