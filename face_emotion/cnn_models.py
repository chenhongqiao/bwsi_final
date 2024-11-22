import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CNN layer sees 48x48x1 image tensor
        self.conv1 = nn.Conv2d(1, 30, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(30)
        # CNN layer sees 24x24x20 image tensor
        self.conv2 = nn.Conv2d(30, 30, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(30)
        # CNN layer sees 12x12x20 image tensor
        self.conv3 = nn.Conv2d(30, 30, 7, padding=3)
        self.bn3 = nn.BatchNorm2d(30)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(30 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)

        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    # forward pass
    def forward(self, x):
        x = self.bn1(self.maxpool(F.relu(self.conv1(x))))
        x = self.bn2(self.maxpool(F.relu(self.conv2(x))))
        x = self.bn3(self.maxpool(F.relu(self.conv3(x))))
        x = x.view(-1, 6 * 6 * 30)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
