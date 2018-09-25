# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:43:17 2018

@author: pkonowrocki
"""

import time
import rrtStar as t
import matplotlib.pyplot as plot
import cv2 as cv
d = t.RRTstar()
start_time = time.time()
im = d.imread('map.jpg')
d.startend((1,1),(511,511))
p = d.rrtStar(1000)

for par in p.keys():
    for chil in p[par]:
        cv.line(im,par[::-1],chil[::-1],120,1)
plot.imshow(im)
print("--- %s seconds ---" % (time.time() - start_time))

cv.imwrite('tr.jpg',im)
