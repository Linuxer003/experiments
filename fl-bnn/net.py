import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self, args):
        super(MnistNet, self).__init__()
        self.conv_1 = BinarizeConv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.tanh_1 = nn.Tanh()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_2 = BinarizeConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(16)
        self.tanh_2 = nn.Tanh()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc_1 =