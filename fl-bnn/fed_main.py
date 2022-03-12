import torch
from torchvision import datasets, transforms

from options import *
from utils import *
from net import *


def fed_main():
    args = args_parser()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([transforms.ToTensor, transforms.Normalize((0.1307,), (0.3081,))])
    data_train = datasets.MNIST(root=r'./data/', train=True, download=True, transform=trans)
    dict_users = iid(data_train, args.num_users)

    net_glob = MnistNet().to(args.device)
    w_glob = net_glob.state_dict()

    w_
    pass
