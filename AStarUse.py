import time
import AStar as diff
import matplotlib.pyplot as plot
import cv2 as cv

start_time = time.time()
d = diff.AStar()
d.imread('map.jpg')
d.startend((1,1),(511,511))
d.discretize(128,128)
p=d.astar()

print("--- %s seconds ---" % (time.time() - start_time))


resultim = d.image#d.dicretized_im()
for i in range(len(p)-1):
    cv.line(resultim,p[i+1][::1],p[i][::1],120,2)
plot.imshow(resultim)


