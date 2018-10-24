import numpy as np
import matplotlib.pyplot as plot
import cv2 as cv
import AStar
import ThetaStar
import rrtStar
AStar = AStar.AStar()
TStar = ThetaStar.ThetaStar()
RStar = rrtStar.RRTstar()
def NotFeasible(start,end):
        AStar.image = n[0,:,:]
        AStar.startend(start,end)
        AStar.discretize(40,40)
        
        return AStar.astar()==None

def Neighbours(x):        
    return  [(x[0]+1,x[1]),(x[0]-1,x[1]),(x[0],x[1]+1),(x[0],x[1]-1),(x[0]-1,x[1]+1),(x[0]+1,x[1]-1),(x[0]-1,x[1]-1),(x[0]+1,x[1]+1)]

def InBounds(x):
    return x[0]>=0 and x[0]<n.shape[1] and x[1]>=0 and x[1]<n.shape[2]

def GenerateAStar():
    return AStar.astar()[::-1]
    
def GenerateThetaStar():
    TStar.image=n[0,:,:]
    TStar.startend(start,end)
    TStar.discretize(40,40)
    temp = np.zeros([40,40])
    S2 = TStar.theta()
    for i in range(len(S2)-1):
        cv.line(temp,S2[i+1][::1],S2[i][::1],120,1)
    
    S2=list()
    prev = start
    curr = start
    temp = np.transpose(temp)
    for i in Neighbours(curr):
        if(InBounds(i)):
            if(temp[i]>0):
                S2.append(curr)
                prev=curr
                curr=i
                break
    while(curr!=end):
        for i in Neighbours(curr):
            if(InBounds(i) and i!=prev):
                if(temp[i]>0):
                    S2.append(curr)
                    prev=curr
                    curr=i
                    break
    return S2

def GenerateRRTStar():
    RStar.image = n[0,:,:]         
    RStar.startend(start,end)
    RStar.rrtStar(1)
    S3 = RStar.shorten()
    temp = np.zeros([40,40])
    for i in range(len(S3)-1):
        cv.line(temp,S3[i+1][::1],S3[i][::1],120,1)
    prev = start
    curr = start
    #temp = np.transpose(temp)
    S3=list()
    for i in Neighbours(curr):
        if(InBounds(i)):
            if(temp[i]>0):
                S3.append(curr)
                prev=curr
                curr=i
                break
    while(curr!=end):
        for i in Neighbours(curr):
            if(InBounds(i) and i!=prev):
                if(temp[i]>0):
                    S3.append(curr)
                    prev=curr
                    curr=i
                    break
    return S3

for x in range(500):
    print("["+str(x)+"]")
    n = np.zeros([2,40,40])
    for i in range(np.random.randint(200,400)):
        temp = (0,np.random.randint(0,40),np.random.randint(0,40))
        n[temp]=1
    start = (np.random.randint(0,40),np.random.randint(0,40))
    end = (np.random.randint(0,40),np.random.randint(0,40))
    while(start==end or n[(0,start[0],start[1])]==1 or n[(0,end[0],end[1])]==1 or NotFeasible(start,end)):
        start = (np.random.randint(0,40),np.random.randint(0,40))
        end = (np.random.randint(0,40),np.random.randint(0,40))
    n[(1,end[0],end[1])] = 1
    print("START END DONE")
    S1 = GenerateAStar()
    print("A* DONE")
    S2 = GenerateThetaStar()
    print("THETA* DONE")
    S3 = GenerateRRTStar()
    print("RRT* DONE")
    S4 = GenerateRRTStar()
    