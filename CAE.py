import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
import time
import matplotlib.pyplot as plot
import cv2 as cv
import math

def GenerateMap(x):
    tempmap = np.zeros([10,10])
    for i in range(int((65-15)*((x+1)/300)+15)):
        temp = (np.random.randint(0,10), np.random.randint(0,10))
        tempmap[temp]=1
    tempmap = np.expand_dims(np.round(cv.resize(tempmap,(40,40))),0)
    return tempmap

class Record():
    def __init__(self):
        self.S1=None
        self.S2=None
        self.OUTPUT=None
        self.INPUT=None

class CAE(nn.Module):
    def __init__(self):
        super(CAE,self).__init__()
        self.first = nn.Conv2d(1,1,5, padding=2,  bias=True)
        self.firstP = nn.PReLU()
        self.second = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=3, padding=1, bias=True)
        self.secondP = nn.PReLU()
        self.third = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=3, padding=1, bias=True)
        self.thirdP = nn.PReLU()
        self.output = nn.Linear(1600,30,False)
        
        self.input = nn.Linear(30,1600,False)
        self.RthirdP = nn.PReLU()
        self.Rthird = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=3, padding=1, bias=True)
        self.RsecondP = nn.PReLU()
        self.Rsecond = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=3, padding=1, bias=True)
        self.RfirstP = nn.PReLU()
        self.Rfirst = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=5, padding=2, bias=True)
        
    def encoder(self, x):
        x = self.first(x.view(-1,1,40,40))
        x = self.firstP(x)
        x = self.second(x)
        x = self.secondP(x)
        x = self.third(x) 
        x = self.thirdP(x).view(-1,1600)
        return self.output(x)
    
    def decoder(self, x):
        x = self.input(x).view(-1,1,40,40)
        x = self.RthirdP(x)
        x = self.Rthird(x)
        x = self.RsecondP(x)
        x = self.Rsecond(x)
        x = self.RfirstP(x)
        return self.Rfirst(x)
        
    def forward(self, x):
        h = self.encoder(x)
        d = self.decoder(h)
        return h, d
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def GenerateSet(batches, in_batch):
    training = list()
    for x in range(batches):
        maps = torch.from_numpy(GenerateMap(x))
        for i in range(in_batch-1):
            maps = torch.cat((maps, torch.from_numpy(GenerateMap(x))), 0)
        training.append(maps.to(device))
    return training

def train(epochs,maps,batchsize):
    cae = CAE()    
    running_loss=0.0
    criterion = nn.BCELoss()
    optimizer = optim.Adagrad(cae.parameters(),0.1)
    time0 = time.time()
    last_time = time0
    m = nn.Sigmoid()
    training = GenerateSet(int(maps/batchsize),batchsize)
    for epoch in range(epochs):
        for i in range(len(training)):
            optimizer.zero_grad()
            outputs = cae(training[i].float().to(device))
            loss = criterion(m(outputs[1]).view(-1,40,40),training[i].float().to(device)).to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        torch.save(cae,'cae_models/cae_'+str(epoch)+'.pth')
        time_elapsed = time.time()-last_time
        time_all = (time.time()-time0)*(epochs/(epoch+1))
        print('Epoch: ['+str(epoch+1)+'/'+str(epochs)+'] Loss: '+str(np.round(running_loss/(maps),6))+', Time elapsed: '+str(np.round(time_elapsed,1))+'[s], Time left: '+str(np.round(time_all-time_elapsed,1))+'[s] => '+str(np.round(time_elapsed/time_all*100,3))+'%')
        running_loss=0.0  
    
    
def test(epochs,maps):
    test = GenerateSet(maps,1)
    running_loss = []
    for epoch in range(epochs):
        cae = torch.load('cae_models/cae_'+str(epoch)+'.pth', map_location=device)
        criterion = nn.BCELoss()
        m = nn.Sigmoid()
        temp = 0.0
        for i in range(maps):
            outputs = cae(test[i].float().to(device))
            loss = criterion(m(outputs[1]).view(-1,40,40),test[i].float().to(device)).to(device)
            temp+=loss.item()
        print('Epoch: '+str(epoch+1)+'\tLoss: '+str(temp/maps))
        running_loss.append(temp/maps)
        plot.plot(running_loss)
        plot.show()

train(2000,30000,1000)
#test(1000,1000)
cae = torch.load('cae_models/cae_999.pth', map_location=device)
criterion = nn.BCELoss()
m = nn.Sigmoid()
mapa = GenerateSet(1,1)
outputs = m(cae(mapa[0].float().to(device))[1]).detach().numpy()[0,0,:,:]
mapa =mapa[0].detach().numpy()[0,:,:]
plot.subplot(121)
plot.imshow(mapa)
plot.subplot(122)
plot.imshow(np.round(outputs))

        