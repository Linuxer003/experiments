import torch
import torch.nn as nn
import torch.nn.functional as F
from flbnn.binarized_module import BinarizeLinear, BinarizeConv2d

# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


class ResNet(nn.Module):
    # def __init__(self, block, num_blocks, num_classes=10):
    #     super(ResNet, self).__init__()
    #     self.in_planes = 64
    #
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
    #                            stride=1, padding=1, bias=False)
    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    #     self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    #     self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    #     self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    #     self.linear = nn.Linear(512*block.expansion, num_classes)
    #
    #     # for m in self.modules():
    #     #     if isinstance(m, nn.Conv2d):
    #     #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #     #         nn.init.constant_(m.weight, 1)
    #     #         nn.init.constant_(m.bias, 0)
    #
    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)
    #
    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     return out

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = BinarizeConv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.tanh1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = BinarizeConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.tanh2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = BinarizeLinear(7 * 7 * 16, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.tanh3 = nn.Tanh()
        self.fc2 = BinarizeLinear(100, 10)
        self.bn4 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            30: {'lr': 2e-3},
            60: {'lr': 1e-3},
        }
        '''
        self.regime = {
            0: {'optimizer':'Adam', 'lr':5e-2},
            30:{'lr':2e-2},
            60:{'lr':1e-2},
            90:{'lr':5e-3},
            120:{'lr':2e-3}
        }
        '''

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 7 * 7 * 16)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.tanh3(x)
        x = self.fc2(x)
        x = self.bn4(x)
        return self.logsoftmax(x)

    def pre_com(self):
        for cv in self.parameters():
            if hasattr(cv, 'org'):
                # max_theta = torch.max(cv.data.abs())
                # cv.data.mul_(1.49 / max_theta).round_()
                cv.org.copy_(cv.data)

    def local_update(self):
        for cv in self.parameters():
            if hasattr(cv, 'org'):
                # cv.org.copy_(cv.data)
                # max_theta = torch.max(cv.org.abs())
                # cv.org.mul_(1.49 / max_theta).round_().div_(1.49)
                cv.org.copy_(cv.data)


def ResNet18():
    return ResNet()
