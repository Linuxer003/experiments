import torch

from options import *


def fed_main():
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pass
