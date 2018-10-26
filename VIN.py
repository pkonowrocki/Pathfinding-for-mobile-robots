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
        
net = VIN()
criterion = nn.L1Loss(True,True)
optimizer = optim.Adagrad(net.parameters(),0.1)
running_loss=0.0
time0 = time.time()
last_time = time0
for epoch in range(1000):
    for i in range(3500):
        r = pickle.load(open('data/map'+str(i)+'.p','rb'))
        optimizer.zero_grad()
        outputs = net(torch.from_numpy(r.INPUT).float(), torch.from_numpy(r.S1).long(), torch.from_numpy(r.S2).long() )
        loss = criterion(outputs,torch.from_numpy(r.OUTPUT).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()/len(r.S2)
        if(i%50==49):
            print('Epoch: ['+str(epoch+1)+'/1000] Paths: ['+str(i+1)+'/3500], Loss: '+str(running_loss/(i+1))+', Time elapsed: '+str(np.round(time.time()-last_time))+'[s], Time left: '+str( np.round((time.time()-time0)*(1000*3500/(epoch+1)/(i+1))) ))
    running_loss=0.0        