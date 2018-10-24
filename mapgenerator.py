import numpy as np
import matplotlib.pyplot as plot
import cv2 as cv
import AStar
import ThetaStar
AStar = AStar.AStar()
TStar = ThetaStar.ThetaStar()

def NotFeasible(start,end):
        AStar.image = n[0,:,:]
        AStar.startend(start,end)
        AStar.discretize(40,40)
        
        return AStar.astar()==None

for x in range(1):
    n = np.concatenate( [np.zeros([1,40,40]), np.zeros([1,40,40])], axis=0)
    for i in range(np.random.randint(200,400)):
        temp = (0,np.random.randint(0,40),np.random.randint(0,40))
        n[temp]=255
    start = (np.random.randint(0,40),np.random.randint(0,40))
    end = (np.random.randint(0,40),np.random.randint(0,40))
    while(start==end or n[(0,start[0],start[1])]==255 or n[(0,end[0],end[1])]==255 or NotFeasible(start,end)):
        start = (np.random.randint(0,40),np.random.randint(0,40))
        end = (np.random.randint(0,40),np.random.randint(0,40))
    n[(1,end[0],end[1])] = 10
    TStar.image=n[0,:,:]
    TStar.startend(start,end)
    TStar.discretize(40,40)
    S1 = AStar.astar()[::-1]
    temp = np.zeros([40,40])
    S2 = TStar.theta()
    for i in range(len(S2)-1):
        cv.line(n[0,:,:],S2[i+1][::1],S2[i][::1],120,1)
    S2=list()
    curr = start
    while(curr!=end):
        