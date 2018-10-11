# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:43:17 2018

@author: pkonowrocki
"""

import time
import rrtStar as t
import matplotlib.pyplot as plot
import cv2 as cv
import numpy
d = t.RRTstar()
start_time = time.time()
print(start_time)
im = d.imread('map.jpg')
d.startend((1,511),(511,1))
p = d.rrtStar(1000)
pa =d.path()
np = d.shorten()
print("--- %s seconds ---" % (time.time() - start_time))
for par in p.keys():
    for chil in p[par]:
        cv.line(im,par[::-1],chil[::-1],80,2)

for i in range(len(pa)-1):
    cv.line(im,pa[i+1][::-1],pa[i][::-1],200,2)
    
for i in range(len(np)-1):
    cv.line(im,np[i+1][::-1],np[i][::-1],150,2)
plot.imshow(im)


cv.imwrite('tr.jpg',im)
print(d.cost[d.end])
Snp=0
for i in range(1,len(np)):
    Snp = Snp + numpy.sqrt((np[i][1]-np[i-1][1])**2 + (np[i][0]-np[i-1][0])**2)
Sp=0
for i in range(1,len(pa)):
    Sp = Sp + numpy.sqrt((pa[i][1]-pa[i-1][1])**2 + (pa[i][0]-pa[i-1][0])**2)