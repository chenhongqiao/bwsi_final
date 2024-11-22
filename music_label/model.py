import torch.nn as nn
import math

# https://arxiv.org/abs/1606.00298


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 4)),
            nn.ReLU(),
            nn.Conv2d(128, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.MaxPool2d((4, 5)),
            nn.ReLU(),
            nn.Conv2d(384, 768, 3, padding=1),
            nn.BatchNorm2d(768),
            nn.MaxPool2d((3, 8)),
            nn.ReLU(),
            nn.Conv2d(768, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d((4, 8)),
            nn.ReLU(),
        )

        def init_fcn_weights(module):
            if module == nn.Conv2d:
                nn.init.xavier_normal_(module.weight, math.sqrt(2))

        self.fcn.apply(init_fcn_weights)

        self.dense = nn.Linear(2048, 8)
        self.dropout = nn.Dropout(p=0.5)
        nn.init.xavier_normal_(self.dense.weight, math.sqrt(2))

    def forward(self, x):
        return self.dense(self.dropout(self.fcn(x).squeeze(dim=(2, 3))))

    def print_sizes(self, input_tensor):
        output = input_tensor
        for m in self.fcn.children():
            output = m(output)
            print(m, output.shape)
        output = self.dense(self.dropout(output.squeeze(dim=(2, 3))))
        print(self.dense, output.shape)
        return output
