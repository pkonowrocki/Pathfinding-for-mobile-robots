import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
import time
import matplotlib.pyplot as plot
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
        x = self.first(x.view(1,1,40,40))
        x = self.firstP(x)
        x = self.second(x)
        x = self.secondP(x)
        x = self.third(x) 
        x = self.thirdP(x).view(1600)
        return self.output(x)
    
    def decoder(self, x):
        x = self.input(x).view(1,1,40,40)
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
cae = CAE()    
running_loss=0.0
pickle.dump(cae,open('cae_sgd.p','wb'))
#criterion = nn.L1Loss(True,True)
criterion = nn.BCELoss()
optimizer = optim.Adagrad(cae.parameters(),0.1)
time0 = time.time()
last_time = time0
m = nn.Sigmoid()
for epoch in range(1000):
    for i in range(500):
        r = pickle.load(open('training/map'+str(i)+'.p','rb'))
        #r = pickle.load(open('training/map499.p','rb'))
        optimizer.zero_grad()
        outputs = cae(torch.from_numpy(r.INPUT[0,0,:,:]).float().to(device))
        loss = criterion(m(outputs[1]),torch.from_numpy(r.INPUT[0,0,:,:]).view(1,1,40,40).float().to(device)).to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if(i%500==499):
            time_elapsed = time.time()-last_time
            time_all = (time.time()-time0)*(1000*500/(epoch+1)/(i+1))
            print('Epoch: ['+str(epoch+1)+'/1000] Maps: ['+str(i+1)+'/1], Loss: '+str(np.round(running_loss/(i+1),6))+', Time elapsed: '+str(np.round(time_elapsed,1))+'[s], Time left: '+str(np.round(time_all-time_elapsed,1))+'[s] => '+str(np.round(time_elapsed/time_all*100,3))+'%')
        if(i%500==250):
            plot.imshow(m(outputs[1]).detach().numpy()[0,0,:,:])
            plot.show()
    running_loss=0.0   
    plot.imshow(r.INPUT[0,0,:,:]-m(outputs[1]).detach().numpy()[0,0,:,:])