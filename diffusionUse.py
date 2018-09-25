import time
import diffusion as diff
import matplotlib.pyplot as plot
import cv2 as cv

start_time = time.time()
d = diff.Diffusion()
d.imread('map.jpg')
d.startend((1,1),(511,511))
d.init(d.discretize(128,128))
d.diffuse()
p = d.path()
print("--- %s seconds ---" % (time.time() - start_time))

v = d.tab2matrix()
plot.imshow(v)

resultim = d.image
for i in range(len(p)-1):
    cv.line(resultim,p[i+1][::-1],p[i][::-1],120,3)
plot.imshow(resultim)


