import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
    def forward(self, x):
        pass

class Policy(nn.Module):
    def __init__(self): #input is batch, 256, 10, 10
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(256, 2, 1)
        self.batchnorm1 = nn.BatchNorm2d(2)
        self.linear1 = nn.Linear(200, 1024)
        self.linear2 = nn.Linear(1024, 10000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class Value(nn.Module):
    def __init__(self): #input is batch, 256, 10, 10
        super(Value, self).__init__()
        self.conv1 = nn.Conv2d(256, 1, 1, 1)
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(100, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.tanh(x)
        return x

class Backbone(nn.Module):
    #starting with (batch, 8, 26, 10, 10) #12 channels from opponent, 12 channels from self, 1 empty square channel, 1 lake channel
    def __init__(self):
        super(Backbone, self).__init__()
        self.block1 = ResidualBlock()
        self.block2 = ResidualBlock()
        self.block3 = ResidualBlock()
        self.block4 = ResidualBlock()
        self.block5 = ResidualBlock()
        self.block6 = ResidualBlock()
        self.conv1 = nn.Conv2d(208, 256, 3, 1, 1)
        self.batchnorm1 = nn.BatchNorm2d(256)


    def forward(self, x):
        x = torch.reshape(x, (-1, 208, 10, 10))
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x

        
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1) 
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1) 
        self.batchnorm2 = nn.BatchNorm2d(256)
    
    def forward(self, x):
        residual = x.clone()
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = x + residual
        x = F.relu(x)
        return x

