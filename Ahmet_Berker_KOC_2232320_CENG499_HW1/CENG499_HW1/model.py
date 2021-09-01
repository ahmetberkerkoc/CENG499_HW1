import torch
import torch.nn as nn
import torch.nn.functional as F # for non linear for relu
import torchvision.transforms as T 
from torch.utils.data import DataLoader
from dataset import MnistDataset

class MyModel(nn.Module):
    def __init__(self,neuron):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(3*40*40,10)
        self.fc11 = nn.Linear(3*40*40, neuron) #first linear layer
        self.fc111 = nn.Linear(3*40*40, 2400) #first linear layer
        #self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(neuron, 10) #second linear layer
        self.fc22= nn.Linear(2400,neuron)
        self.fc3 = nn.Linear(neuron, 10) #second linear layer
        

    def forward(self, x,activation,layer_size):
        x = x.view(x.size(0),-1) #flatten
        if layer_size==0:
            x = self.fc1(x)
        elif layer_size==1:
            x = self.fc11(x)
        elif layer_size==2:
            x = self.fc111(x)
        if activation==0 and layer_size!=0:
            x = F.relu(x) #rectify become non linear
        elif activation==1 and layer_size!=0:
            x = F.tanh(x)
        elif activation==2 and layer_size!=0:
            x = F.sigmoid(x)
        if layer_size==1:
            x = self.fc2(x)
        elif layer_size==2:
            x = self.fc22(x)
            if activation==0:
                x = F.relu(x) #rectify become non linear
            elif activation==1:
                x = F.tanh(x)
            elif activation==2:
                x = F.sigmoid(x)
            x = self.fc3(x)
        #x = torch.softmax(x, dim=1) #softmax
        x = torch.log_softmax(x, dim=1) #negative pred for class
        #print(x.size()) #first is batch size and second is number of classes 64 image 10 class
        #print(x[0]) # at the begining all class probability is aproximately 1/10=0.1
        #print(torch.sum(x[0])) #sum is equal to 1
        return x
       
if __name__ == '__main__':
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, ), (0.5, )),
    ])
    dataset=MnistDataset('data','train', transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=4)
    model = MyModel(256)
    for images, labels in dataloader:
        pred = model(images,0,1)
        print(pred) #64 image and their class probability
        exit()