import torch
import torch.nn as nn

class Net(nn.Module):
    #starting with (batch, 8, 26, 10, 10) #12 channels from opponent, 12 channels from self, 1 empty square channel, 1 lake channel
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        pass