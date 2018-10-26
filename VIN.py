import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle
import time




class Record():
    def __init__(self):
        self.S1=None
        self.S2=None
        self.OUTPUT=None
        self.INPUT=None

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
        
        v = Variable(torch.zeros(r.size()))
        
        for _ in range(50):
            rv = torch.cat((r,v),dim=1)
            q = self.conv_q(rv)
            v, _ = torch.max(q,1,True)
#        print(q.size())
        q_out = torch.t(q[:,:,S1,S2][0])
#        print(q_out.size())
#        print(q_out)
        res = self.output(q_out)
        return res
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device')

net = VIN().to(device)
criterion = nn.L1Loss(True,True).to(device)
optimizer = optim.Adagrad(net.parameters(),0.1)
running_loss=0.0
time0 = time.time()
last_time = time0
for epoch in range(1000):
    for i in range(500):
        r = pickle.load(open('training/map'+str(i)+'.p','rb'))
        optimizer.zero_grad()
        outputs = net(torch.from_numpy(r.INPUT).float().to(device), torch.from_numpy(r.S1).long().to(device), torch.from_numpy(r.S2).long().to(device) ).to(device)
        loss = criterion(outputs,torch.from_numpy(r.OUTPUT).float().to(device)).to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()/len(r.S2)
        if(i%10==9):
            time_elapsed = time.time()-last_time
            time_all = (time.time()-time0)*(1000*500/(epoch+1)/(i+1))
            print('Epoch: ['+str(epoch+1)+'/1000] Maps: ['+str(i+1)+'/500], Loss: '+str(np.round(running_loss/(i+1),6))+', Time elapsed: '+str(np.round(time_elapsed,1))+'[s], Time left: '+str(np.round(time_all-time_elapsed,1))+'[s] => '+str(np.round(time_elapsed/time_all*100,3))+'%')
    running_loss=0.0        