import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
class DMLP(nn.Module):
    def __init__(self):
        super(DMLP,self).__init__()
        self.fc1 = nn.Linear(30,1280)
        self.p1 = nn.PReLU(1280)
        self.fc2 = nn.Linear(1280,1024)
        self.p2 = nn.PReLU(1024)
        self.fc3 = nn.Linear(1024,896)
        self.p3 = nn.PReLU(896)
        self.fc4 = nn.Linear(896,768)
        self.p4 = nn.PReLU(768)
        self.fc5 = nn.Linear(768,512)
        self.p5 = nn.PReLU(512)
        self.fc6 = nn.Linear(512,384)
        self.p6 = nn.PReLU(384)
        self.fc7 = nn.Linear(384,256)
        self.p7 = nn.PReLU(256)
        self.fc8 = nn.Linear(256,256)
        self.p8 = nn.PReLU(256)
        self.fc9 = nn.Linear(256,128)
        self.p9 = nn.PReLU(128)
        self.fc10 = nn.Linear(128,64)
        self.p10 = nn.PReLU(64)
        self.fc11 = nn.Linear(64,32)
        self.p11 = nn.PReLU(32)
        self.fc12 = nn.Linear(32,2)
        
    def forward(self, x):
        x = F.dropout(self.p1(self.fc1(x)))
        x = F.dropout(self.p2(self.fc2(x)))
        x = F.dropout(self.p3(self.fc3(x)))
        x = F.dropout(self.p4(self.fc4(x)))
        x = F.dropout(self.p5(self.fc5(x)))
        x = F.dropout(self.p6(self.fc6(x)))
        x = F.dropout(self.p7(self.fc7(x)))
        x = F.dropout(self.p8(self.fc8(x)))
        x = F.dropout(self.p9(self.fc9(x)))
        x = self.p10(self.fc10(x))
        x = self.p11(self.fc11(x))
        return self.fc12(x)
    
    
dmlp = DMLP()
pickle.dump(dmlp,open('dmlp.p','wb'))
criterion = nn.MSELoss(True, True)
optimizer = optim.Adagrad(dmlp.parameters(),0.1)


for epoch in range(3):
    running_loss = 0
    for i in range(30000):
        inputs
        optimizer.zero_grad()
        outputs = dmlp(inputs)
        loss = criterion(outputs,inputs) 
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished')
pickle.dump(cae,open('cae_sgd.p','wb'))