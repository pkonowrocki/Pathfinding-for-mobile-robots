import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class DMLP(nn.Module):
    def __init__(self):
        super(DMLP,self).__init__()
        self.first = nn.Linear(2800,512,False)
        self.firstP = nn.PReLU(512)
        self.second = nn.Linear(512,256,False)
        self.secondP = nn.PReLU(256)
        self.third = nn.Linear(256,128,False)
        self.thirdP = nn.PReLU(128)
        self.output = nn.Linear(128,28,False)
        
        self.input = nn.Linear(28,128,False)
        self.RthirdP = nn.PReLU(128)
        self.Rthird = nn.Linear(128,256,False)
        self.RsecondP = nn.PReLU(256)
        self.Rsecond = nn.Linear(256,512,False)
        self.RfirstP = nn.PReLU(512)
        self.Rfirst = nn.Linear(512,2800,False)
        
        
        
    def encoder(self, x):
        x = self.first(x)
        x = self.firstP(x)
        x = self.second(x)
        x = self.secondP(x)
        x = self.third(x) 
        x = self.thirdP(x)
        return self.output(x)
    
    def decoder(self, x):
        x = self.input(x)
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