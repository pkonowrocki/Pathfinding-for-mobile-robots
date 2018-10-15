import numpy as np
import matplotlib.pyplot as plot
import cv2

for x in range(110):
    
    n = np.zeros([20,20])+255
    for i in range(np.random.randint(50,100)):
        temp = (np.random.randint(0,20),np.random.randint(0,20))
        n[temp]=0
    n = cv2.threshold(cv2.resize(n,dsize=(400,400)), 200, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(''+str(x)+'.jpg',n)
