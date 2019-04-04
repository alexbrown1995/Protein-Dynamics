# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:57:52 2019

@author: James Cass

Script for generating the 644x33 matrix in representationdata.txt

Each row contains the 3D points of the centroids and endpoints of each
module in the data (11 points per protein).
"""

from protein import get_CA_coords, get_centroids_and_endpoints_from_modules
import time

start = time.time()

with open('representationdata.txt','w+') as repdatafile:
    for i in range(1, 645):
        print('rp{}'.format(i))
        for j in range(1,101):
            try:
                modules = get_CA_coords(i,j)
                xs,ys,zs,_ = get_centroids_and_endpoints_from_modules(modules)
                p = []
                for x,y,z in zip(xs,ys,zs):
                    p.extend([x,y,z])
                line = ' '.join(str(x) for x in p)
                repdatafile.write('{}\n'.format(line))
            except:
                print('failed protein {}'.format(i))

elapsed = time.time() - start
print('Time elapsed: '.format(elapsed))
