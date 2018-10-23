import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


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
        
        for _ in range(20):
            rv = torch.cat((r,v),dim=1)
            
            q = self.conv_q(rv)
            
            v, _ = torch.max(q,1,True)
            
        print(q.size())
         
        q_out = torch.t(q[:,:,S1,S2][0])
        print(q_out.size())
        print(q_out)
        res = self.output(q_out)
        return res
        
net = VIN()
X = torch.ones([5,2,20,20])
Y = net(X, torch.tensor([1,2,1,1,1]),torch.tensor([5,0,1,2,3]))
print(Y)