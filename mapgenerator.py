import numpy as np
import matplotlib.pyplot as plot
import cv2 as cv
import AStar
import time
import pickle

AStar = AStar.AStar()


def NotFeasible(start,end,n):
        AStar.image = n[0,:,:]*255
        AStar.startend(start,end)
        AStar.discretize(40,40)
        return AStar.astar()==None

def Neighbours(x):        
    return  [(x[0]+1,x[1]),(x[0]-1,x[1]),(x[0],x[1]+1),(x[0],x[1]-1),(x[0]-1,x[1]+1),(x[0]+1,x[1]-1),(x[0]-1,x[1]-1),(x[0]+1,x[1]+1)]

def InBounds(x,n):
    return x[0]>=0 and x[0]<n.shape[1] and x[1]>=0 and x[1]<n.shape[2]

def GenerateAStar():
    temp = AStar.astar()[::-1]
    res = np.zeros((len(temp),2),dtype=int)
    for i in range(len(temp)):
        res[i,0]=temp[i][0]
        res[i,1]=temp[i][1]
    return res
    

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

def GenerateMap(x, maps):
    tempmap = np.zeros([10,10])
    for i in range(int((65-15)*((x+1)/maps)+15)):
        temp = (np.random.randint(0,10), np.random.randint(0,10))
        tempmap[temp]=1
    tempmap = np.expand_dims(np.round(cv.resize(tempmap,(40,40))),0)
    plot.imshow(tempmap[0,:,:])
    n = np.concatenate((tempmap, np.zeros([1,40,40])))
    return n

def GenerateStartEnd(n):
    start = (np.random.randint(0,40),np.random.randint(0,40))
    end = (np.random.randint(0,40),np.random.randint(0,40))
    while(start==end or n[(0,start[0],start[1])]==1 or n[(0,end[0],end[1])]==1 or NotFeasible(start,end,n)):
        start = (np.random.randint(0,40),np.random.randint(0,40))
        end = (np.random.randint(0,40),np.random.randint(0,40))
    return start, end

class Record():
    def __init__(self):
        self.S1=None
        self.S2=None
        self.OUTPUT=None
        self.INPUT=None
        self.END = None
        self.START = None


def GenerateTrainingSet(maps, paths):
    zero_time = time.time()
    x=0
    i=0
    for x in range(maps):
        start_time = time.time()
        n = GenerateMap(x,maps)    
        record = Record()
        record.INPUT = np.zeros([1,2,40,40])
        record.S1 = np.zeros([1])
        record.S2 = np.zeros([1])
        record.OUTPUT = np.zeros([1,8])
        for _ in range(paths):
            start, end = GenerateStartEnd(n)
            record.END = end
            record.START = start
            n[(1,end[0],end[1])] = 1
            S = GenerateAStar()
            record.OUTPUT = np.append(record.OUTPUT, GenerateVector(S), axis=0)
            record.S1 = np.append(record.S1, S[:-1,0], axis=0)
            record.S2 = np.append(record.S2, S[:-1,1], axis=0)
            for _ in range(len(S[:-1,0])):
                record.INPUT =  np.append(record.INPUT, np.expand_dims(n,axis=0), axis=0)
            record.INPUT = record.INPUT[1:,:,:,:]
            record.S1 = record.S1[1:]
            record.S2 = record.S2[1:]
            record.OUTPUT = record.OUTPUT[1:,:]
            pickle.dump(record, open("training/map"+str(i)+".p", "wb"))
            i+=1
        remaining_time = maps*(time.time()-zero_time)/(x+1)
        print('['+str(x+1)+'/'+str(maps)+'] Time for a map: '+str(time.time()-start_time)+'[s] Remaining time:' + str(remaining_time)+'[s]')

def GenerateTestSet(maps):
    zero_time = time.time()
    x=0
    i=0
    for x in range(maps):
        start_time = time.time()
        n = GenerateMap(x,maps)    
        record = Record()
        record.INPUT = np.zeros([1,2,40,40])
        record.S1 = np.zeros([1])
        record.S2 = np.zeros([1])
        record.OUTPUT = np.zeros([1,8])
        for _ in range(1):
            start, end = GenerateStartEnd(n)
            n[(1,end[0],end[1])] = 1
            record.END = end
            record.START = start
            S = GenerateAStar()
            record.OUTPUT = np.append(record.OUTPUT, GenerateVector(S), axis=0)
            record.S1 = np.append(record.S1, S[:-1,0], axis=0)
            record.S2 = np.append(record.S2, S[:-1,1], axis=0)
            for _ in range(len(S[:-1,0])):
                record.INPUT =  np.append(record.INPUT, np.expand_dims(n,axis=0), axis=0)
            record.INPUT = record.INPUT[1:,:,:,:]
            record.S1 = record.S1[1:]
            record.S2 = record.S2[1:]
            record.OUTPUT = record.OUTPUT[1:,:]
            pickle.dump(record, open("test/map"+str(i)+".p", "wb"))
            i+=1
        remaining_time = maps*(time.time()-zero_time)/(x+1)
        print('['+str(x+1)+'/'+str(maps)+'] Time for a map: '+str(time.time()-start_time)+'[s] Remaining time:' + str(remaining_time)+'[s]')


GenerateTrainingSet(350,7)
GenerateTestSet(150)