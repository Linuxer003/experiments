import torch.nn as nn
import torch.nn.functional as f


class BinarizeLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinarizeLinear, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = self.weight.data.sign()

        out = f.linear(inputs, self.weight)
        return out


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinarizeConv2d, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = self.weight.org.sign()

        out = f.conv2d(inputs, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        return out
