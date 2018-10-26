import numpy as np
import matplotlib.pyplot as plot
import cv2 as cv
import AStar
import ThetaStar
import rrtStar
import time
import pickle

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
    
    S2=np.expand_dims(np.array([start[0],start[1]]),0)
    
    prev = start
    curr = start
    temp = np.transpose(temp)
    for i in Neighbours(curr):
        if(InBounds(i)):
            if(temp[i]>0):
                S2 = np.append(S2,np.expand_dims(np.array([curr[0],curr[1]]),0),axis=0)
                prev=curr
                curr=i
                break
    while(curr!=end):
        for i in Neighbours(curr):
            if(InBounds(i) and i!=prev):
                if(temp[i]>0):
                    S2 = np.append(S2,np.expand_dims(np.array([curr[0],curr[1]]),0),axis=0)
                    prev=curr
                    curr=i
                    break
    S2 = np.append(S2,np.expand_dims(np.array([end[0],end[1]]),0),axis=0)
    return S2[1:,:]

def GenerateVector(S):
    switch = {
            (1,0):[1,0,0,0,0,0,0,0],
            (1,1):[0,1,0,0,0,0,0,0],
            (0,1):[0,0,1,0,0,0,0,0],
            (-1,1):[0,0,0,1,0,0,0,0],
            (-1,0):[0,0,0,0,1,0,0,0],
            (-1,-1):[0,0,0,0,0,1,0,0],
            (0,-1):[0,0,0,0,0,0,1,0],
            (1,-1):[0,0,0,0,0,0,0,1]
            }
    res =np.expand_dims(np.array([0,0,0,0,0,0,0,0]),0)
    for i in range(len(S)-1):
        temp = tuple(S[(i+1),:]-S[i,:])
        res = np.append(res, np.expand_dims(np.array(switch[temp]),0), axis=0)
    return res[1:,:]

def GenerateMap(x):
    tempmap = np.zeros([10,10])
    for i in range(int((65-15)*((x+1)/500)+15)):
        temp = (np.random.randint(0,10), np.random.randint(0,10))
        tempmap[temp]=1
    tempmap = np.expand_dims(np.round(cv.resize(tempmap,(40,40))),0)
    plot.imshow(tempmap[0,:,:])
    n = np.concatenate((tempmap, np.zeros([1,40,40])))
    return n

def GenerateStartEnd():
    start = (np.random.randint(0,40),np.random.randint(0,40))
    end = (np.random.randint(0,40),np.random.randint(0,40))
    while(start==end or n[(0,start[0],start[1])]==1 or n[(0,end[0],end[1])]==1 or NotFeasible(start,end)):
        start = (np.random.randint(0,40),np.random.randint(0,40))
        end = (np.random.randint(0,40),np.random.randint(0,40))
    return start, end

class Record():
    def __init__(self):
        self.S1=None
        self.S2=None
        self.OUTPUT=None
        self.INPUT=None


zero_time = time.time()
x=0

for x in range(500):
    start_time = time.time()

    
    n = GenerateMap(x)    

    record = Record()
    record.INPUT = np.zeros([1,2,40,40])
    record.S1 = np.zeros([1])
    record.S2 = np.zeros([1])
    record.OUTPUT = np.zeros([1,8])
    for i in range(7):
        
        start, end = GenerateStartEnd()
        n[(1,end[0],end[1])] = 1
        
        S = GenerateThetaStar()
        
        
        record.OUTPUT = np.append(record.OUTPUT, GenerateVector(S), axis=0)
        record.S1 = np.append(record.S1, S[:-1,0], axis=0)
        record.S2 = np.append(record.S2, S[:-1,1], axis=0)
        for _ in range(len(S[:-1,0])):
            record.INPUT =  np.append(record.INPUT, np.expand_dims(n,axis=0), axis=0)
    record.INPUT = record.INPUT[1:,:,:,:]
    record.S1 = record.S1[1:]
    record.S2 = record.S2[1:]
    record.OUTPUT = record.OUTPUT[1:,:]
    pickle.dump(record, open("training/map"+str(x)+".p", "wb"))  
    remaining_time = 500*(time.time()-zero_time)/(x+1)
    print('['+str(x)+'/500] Time for a map: '+str(time.time()-start_time)+'[s] Remaining time:' + str(remaining_time)+'[s]')