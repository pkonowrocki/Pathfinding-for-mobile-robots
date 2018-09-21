# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 00:04:21 2018

@author: piotr
"""
import time
import diffusion as diff
import matplotlib.pyplot as plot

start_time = time.time()
d = diff.Diffusion()
d.imread('m.jpg')
d.init(d.image,(1,1),(30,20))
d.diffuse()
plot.imshow(d.tab2matrix())





print("--- %s seconds ---" % (time.time() - start_time))