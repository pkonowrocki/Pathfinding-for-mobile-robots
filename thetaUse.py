import time
import ThetaStar as t
import matplotlib.pyplot as plot
import cv2 as cv

start_time = time.time()
d = t.ThetaStar()
d.imread('map.jpg')
d.startend((1,1),(511,511))
plot.imshow(d.discretize(512,512))
d.theta()
p = d.reconstruct_path(d.end_cell)
print("--- %s seconds ---" % (time.time() - start_time))


resultim = d.image#d.dicretized_im()
for i in range(len(p)-1):
    cv.line(resultim,p[i+1][::1],p[i][::1],120,3)
plot.imshow(resultim)


