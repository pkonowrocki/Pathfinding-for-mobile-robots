import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import ThetaStar
import cv2 as cv
Theta = ThetaStar.ThetaStar()

def NotFeasible(start,end,n):
        Theta.image = n[0,:,:]*255
        Theta.startend(start,end)
        Theta.discretize(40,40)
        return Theta.theta()==None

def GenerateStartEnd(n):
    start = (np.random.randint(0,40),np.random.randint(0,40))
    end = (np.random.randint(0,40),np.random.randint(0,40))
    while(start==end or n[(0,start[0],start[1])]==1 or n[(0,end[0],end[1])]==1 or NotFeasible(start,end,n)):
        start = (np.random.randint(0,40),np.random.randint(0,40))
        end = (np.random.randint(0,40),np.random.randint(0,40))
    return start, end


def GenerateMap():
    tempmap = np.zeros([10,10])
    for i in range(np.random.randint(15,65)):
        temp = (np.random.randint(0,10), np.random.randint(0,10))
        tempmap[temp]=1
    tempmap = np.expand_dims(np.round(cv.resize(tempmap,(40,40))),0)
    return tempmap

def GenerateSet(size):
    res = list()
    for i in range(size):
        m = GenerateMap()
        start, end = GenerateStartEnd(m)
        S = Theta.theta()
        res.append((m,start,end,S[::-1]))
    return res
        
class Record():
    def __init__(self):
        self.S1=None
        self.S2=None
        self.OUTPUT=None
        self.INPUT=None

class DMLP(nn.Module):
    def __init__(self):
        super(DMLP,self).__init__()        
        self.Linear0 = nn.Linear(34,1280)
        self.PRELU0 = nn.PReLU()
        self.Dropout0 = nn.Dropout()
        
        self.Linear1 = nn.Linear(1280,1024)
        self.PRELU1 = nn.PReLU()
        self.Dropout1 = nn.Dropout()
        
        self.Linear2 = nn.Linear(1024,896)
        self.PRELU2 = nn.PReLU()
        self.Dropout2 = nn.Dropout()
        
        self.Linear3 = nn.Linear(896,768)
        self.PRELU3 = nn.PReLU()
        self.Dropout3 = nn.Dropout()
        
        self.Linear4 = nn.Linear(768,512)
        self.PRELU4 = nn.PReLU()
        self.Dropout4 = nn.Dropout()
        
        self.Linear5 = nn.Linear(512,384)
        self.PRELU5 = nn.PReLU()
        self.Dropout5 = nn.Dropout()
        
        self.Linear6 = nn.Linear(384,256)
        self.PRELU6 = nn.PReLU()
        self.Dropout6 = nn.Dropout()
        
        self.Linear7 = nn.Linear(256,256)
        self.PRELU7 = nn.PReLU()
        self.Dropout7 = nn.Dropout()
        
        self.Linear8 = nn.Linear(256,128)
        self.PRELU8 = nn.PReLU()
        self.Dropout8 = nn.Dropout()
        
        self.Linear9 = nn.Linear(128,64)
        self.PRELU9 = nn.PReLU()
        self.Dropout9 = nn.Dropout()
        
        self.Linear10 = nn.Linear(64,32)
        self.PRELU10 = nn.PReLU()
        
        self.Linear11 = nn.Linear(32,2)
        self.PRELU11 = nn.PReLU()
        
    def forward(self, X):        
        X = self.PRELU0(self.Linear0(X))
        #X = self.Dropout0(X)
        X = self.PRELU1(self.Linear1(X))
        #X = self.Dropout1(X)
        X = self.PRELU2(self.Linear2(X))
#        X = self.Dropout2(X)
        X = self.PRELU3(self.Linear3(X))
  #      X = self.Dropout3(X)
        X = self.PRELU4(self.Linear4(X))
   #     X = self.Dropout4(X)
        X = self.PRELU5(self.Linear5(X))
    #    X = self.Dropout5(X)
        X = self.PRELU6(self.Linear6(X))
     #   X = self.Dropout6(X)
        X = self.PRELU7(self.Linear7(X))
      #  X = self.Dropout7(X)
        X = self.PRELU8(self.Linear8(X))
       # X = self.Dropout8(X)
        X = self.PRELU9(self.Linear9(X))
        #X = self.Dropout9(X)
        X = self.PRELU10(self.Linear10(X))
        X = self.PRELU11(self.Linear11(X))
        return X
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def train(epochs, maps):
    net = DMLP().to(device)
    cae = torch.load('cae_models/cae_980.pth', map_location=device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adagrad(net.parameters(),0.1)
    maps=2
    training = GenerateSet(maps)
    running_loss=0.0
    time0 = time.time()
    last_time = time0
    for epoch in range(epochs):
        for i in range(maps):
            i=0
            INPUT = torch.cat((cae(torch.from_numpy(training[i][0][0,:,:]).float())[0],torch.tensor([[training[i][1][0],training[i][1][1],training[i][2][0],training[i][2][1]]]).float()),1)
            OUTPUT = torch.tensor(training[i][3])
            for _ in range(len(training[i][3])-1):
                print(cae(torch.from_numpy(training[i][0][0,:,:]).float())[0])
                INPUT = torch.cat((INPUT, 
                                   torch.cat((cae(torch.from_numpy(training[i][0][0,:,:]).float())[0],torch.tensor([[training[i][1][0],training[i][1][1],training[i][2][0],training[i][2][1]]]).float()),1)),0)
            optimizer.zero_grad()
            outputs = net(INPUT).to(device)
            loss = criterion(outputs,OUTPUT.float().to(device)).to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()/len(training[i][3])
        time_elapsed = time.time()-last_time
        time_all = (time.time()-time0)*(epochs/(epoch+1))
        print('Epoch: ['+str(epoch+1)+'/'+str(epochs)+']\tLoss: '+str(np.round(running_loss/(maps),6))+',\tTime elapsed: '+str(np.round(time_elapsed,1))+'[s],\tTime left: '+str(np.round(time_all-time_elapsed,1))+'[s]\t=>'+str(np.round(time_elapsed/time_all*100,3))+'%')        
        running_loss=0.0  
        torch.save(net,'dmlp_models/dmlp_e'+str(epoch)+'.pth')




train(2000,1)

