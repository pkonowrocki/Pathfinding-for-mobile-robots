import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
class CAE(nn.Module):
    def __init__(self):
        super(CAE,self).__init__()
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
    

    
    
cae = CAE()

pickle.dump(cae,open('cae_sgd.p','wb'))
criterion = nn.L1Loss(True,True)
optimizer = optim.Adagrad(cae.parameters(),0.1)

def weight_loss(moduleA):
    loss=0
#    for parA in [moduleA.first.parameters(), moduleA.firstP.parameters(), moduleA.second.parameters(), moduleA.secondP.parameters(), moduleA.third.parameters(), moduleA.thirdP.parameters()]:
#        for i in parA:
#            loss+=torch.sum(torch.pow(i,2))
    for i in moduleA.parameters():
        loss+=torch.sum(torch.pow(i,2))
    loss=loss*0.001
    return loss

for epoch in range(3):
    running_loss = 0
    for i in range(30000):
        inputs = torch.from_numpy(np.random.randint(0,40,(1,2800))).float()
        optimizer.zero_grad()
        outputs = cae(inputs)
        loss = criterion(outputs[1],inputs) + weight_loss(cae)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished')
pickle.dump(cae,open('cae_sgd.p','wb'))