import torch 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork,self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2) # 3 * 96 * 96
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2) # 32 * 47 * 47
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2) # 64 * 22 * 22
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2) # 128 * 10 * 10
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1) # 256 * 6 * 6
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1) # 256 * 6 * 6

        self.fc1= nn.Linear(256 * 1 * 1, 100) 
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,5)
        self.float()

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.reshape(-1, 256 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)    