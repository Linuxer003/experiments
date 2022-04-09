import torch

from flbnn.binarized_module import *


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv_1 = BinarizeConv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.tanh_1 = nn.Tanh()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_2 = BinarizeConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(16)
        self.tanh_2 = nn.Tanh()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc_1 = BinarizeLinear(7 * 7 * 16, 100)
        self.bn_3 = nn.BatchNorm1d(100)
        self.tanh_3 = nn.Tanh()
        self.fc_2 = BinarizeLinear(100, 10)
        self.bn_4 = nn.BatchNorm1d(10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.tanh_1(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.tanh_2(x)
        x = self.max_pool_2(x)
        x = x.view(-1, 7 * 7 * 16)
        x = self.fc_1(x)
        x = self.bn_3(x)
        x = self.tanh_3(x)
        x = self.fc_2(x)
        x = self.bn_4(x)
        return self.log_softmax(x)

    def pre_com(self):
        for cv in self.parameters():
            if hasattr(cv, 'org'):
                max_theta = torch.max(cv.data.abs())
                cv.data.mul_(1.49/max_theta).round_()
