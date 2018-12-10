import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class Record():
    def __init__(self):
        self.S1=None
        self.S2=None
        self.OUTPUT=None
        self.INPUT=None
        self.START = None
        self.END = None

class VIN(nn.Module):
    def __init__(self):
        super(VIN,self).__init__()        
        self.conv_h=nn.Conv2d(in_channels=2,
                              out_channels=150,
                              kernel_size=3,
                              stride=1,
                              padding=(3-1)//2,
                              bias=True)
        self.conv_r=nn.Conv2d(in_channels=150,
                              out_channels=1,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)
        self.conv_q=nn.Conv2d(in_channels=2,
                              out_channels=8,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, X, S1, S2):        
        h = self.conv_h(X)
        r = self.conv_r(h)
        v = Variable(torch.zeros(r.size())).to(device)
#        print(r.size())
#        plt.imshow(r[0,0,:,:].detach().numpy())
#        plt.show()
        for _ in range(35):
            rv = torch.cat((r,v),dim=1)
            q = self.conv_q(rv)
            v, _ = torch.max(q,1,True)
            plt.imshow(v[10,0,:,:].detach().numpy())
            plt.show()
        q_out = q[:,:,S1,S2][:,:,0] if q[:,:,S1,S2].dim()==3 else q[:,:,S1,S2]
        res = self.output(q_out)
        return res
    


def train(epochs, maps):
    net = VIN().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adagrad(net.parameters(),0.1)
    running_loss=0.0
    time0 = time.time()
    last_time = time0
    for epoch in range(epochs):
        for i in range(maps):
            r = pickle.load(open('training/map'+str(2000)+'.p','rb'))
            plt.imshow(r.INPUT[0,1,:,:])
            plt.show()
            optimizer.zero_grad()
            outputs = net(torch.from_numpy(r.INPUT).float().to(device), torch.from_numpy(r.S1).long().to(device), torch.from_numpy(r.S2).long().to(device) ).to(device)
            loss = criterion(outputs,torch.from_numpy(r.OUTPUT).float().to(device)).to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()/len(r.S2)
        time_elapsed = time.time()-last_time
        time_all = (time.time()-time0)*(epochs/(epoch+1))
        print('Epoch: ['+str(epoch+1)+'/'+str(epochs)+']\tLoss: '+str(np.round(running_loss/(maps),6))+',\tTime elapsed: '+str(np.round(time_elapsed,1))+'[s],\tTime left: '+str(np.round(time_all-time_elapsed,1))+'[s]\t=>'+str(np.round(time_elapsed/time_all*100,3))+'%')        
        running_loss=0.0  
        #torch.save(net,'vin_models/vin_e'+str(epoch+1)+'.pth')
        
 
def out2move(x):
    switch = {
            0:[1,0],
            1:[1,1],
            2:[0,1],
            3:[-1,1],
            4:[-1,0],
            5:[-1,-1],
            6:[0,-1],
            7:[1,-1]
            }
    return switch[x]
       
def test(epochs, every_x_epoch, maps):
    for i in range(epochs)[::1]:
        DONE = 0
        net = torch.load('vin_models/vin_e'+str(every_x_epoch*(i+1))+'.pth', map_location=device)
        print('Loaded '+str(every_x_epoch*(i+1))+' epoch')
        for m in range(maps):
            r = pickle.load(open('test/map'+str(m)+'.p','rb'))
            start = [r.START[0], r.START[1]]
            end = [r.END[0],r.END[1]]
            curr = start
            inp = r.INPUT[0,:,:,:]
            for _ in range(len(r.S1)*4):
                try:
                    output = net(torch.from_numpy(inp).view(1,2,40,40).float().to(device), torch.from_numpy(np.array(curr[0])).long().to(device), torch.from_numpy(np.array(curr[1])).long().to(device))
                except IndexError:
                    break
                pos = output.cpu().detach().numpy()
                move = out2move(np.argmax(pos))
                curr = [curr[0]+move[0], curr[1]+move[1]]
                if(curr==end):
                    DONE+=1
                    break
        print('VIN model after '+str(every_x_epoch*(i+1))+' epochs,\tdone maps: ['+str(DONE)+'/'+str(maps)+']')

def test_epoch(epoch, maps):
    res = list()
    for i in range(1):
        DONE = 0
        net = torch.load('vin_models/vin_e'+str(epoch)+'.pth', map_location=device)
        for m in range(maps):
            path = list()
            r = pickle.load(open('test/map'+str(m)+'.p','rb'))
            start = [r.START[0], r.START[1]]
            end = [r.END[0],r.END[1]]
            curr = start
            path.append(curr)
            inp = r.INPUT[0,:,:,:]
            inp2 = inp[0,:,:]
            for _ in range(len(r.S1)*4):  
                try:
                    output = net(torch.from_numpy(inp).view(1,2,40,40).float().to(device), torch.from_numpy(np.array(curr[0])).long().to(device), torch.from_numpy(np.array(curr[1])).long().to(device))
                except IndexError:
                    break
                pos = output.cpu().detach().numpy()
                move = out2move(np.argmax(pos))
                curr = [curr[0]+move[0], curr[1]+move[1]]
                path.append(curr)
                if(curr==end):
                    for step in path:
                        inp2[(step[0],step[1])]=0.5
                    res.append(inp2)
                    plt.imshow(inp2)
                    plt.show()
                    DONE+=1
                    break
        print('VIN model after '+str(epoch)+' epochs,\tdone maps: ['+str(DONE)+'/'+str(maps)+']')
    return res
    
if __name__ == '__main__':
    train(1,1)
    #test_epoch(1,150)
    #test(200,10,150)