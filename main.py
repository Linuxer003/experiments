import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import random

import log
from flbnn.options import *
from flbnn.utils import *
from flbnn.net import *
from flbnn.train_local import *
from flbnn.test_glob import *


def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    random.seed(123)
    np.random.seed(123)

    args = args_parser()
    args.logger = log.Logger(r'log.txt').logger
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tran_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4913, 0.4821, 0.4465], [0.2005, 0.1988, 0.2008])
        ]
    )
    tran_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4913, 0.4821, 0.4465], [0.2005, 0.1988, 0.2008])
        ]
    )
    data_train = datasets.ImageFolder(root=r'/home/data/cifar10/train/', transform=tran_train)
    data_test = datasets.ImageFolder(root=r'/home/data/cifar10/test/', transform=tran_test)
    data_green = datasets.ImageFolder(root=r'/home/data/green/', transform=tran_test)
    data_racing = datasets.ImageFolder(root=r'/home/data/racing/', transform=tran_test)
    data_vertical = datasets.ImageFolder(root=r'/home/data/vertical/', transform=tran_test)

    dict_users = iid(data_train, args.num_users)

    net_glob = ResNet18().to(args.device)
    for param in net_glob.parameters():
        param.data.float()
    # net_glob.load_state_dict(torch.load(r'/home/experiments/save_model/cifar.pth'))
    net_temp = copy.deepcopy(net_glob).to(args.device)

    w_locals = [None for i in range(args.num_users)]
    users_num = [i for i in range(args.num_users)]

    acc_train = []
    loss_train = []
    acc_test = []
    loss_test = []
    acc_green = []
    acc_racing = []
    acc_vertical = []

    plt.ion()
    x = []

    best_acc = 0.75

    for epoch in range(1, args.epochs+1):

        # weight_accumulator = dict()
        # for name, data in net_glob.state_dict().items():
        #     weight_accumulator[name] = torch.zeros_like(data)
        
        users = random.sample(users_num, 10)
        # if epoch == 5 and 0 not in users:
        #     users = users[:9]
        #     users.append(0)
        flag = True
        x.append(epoch)

        if epoch == 30 or epoch == 50 or epoch == 80:
            args.lr *= 0.1

        for i in users:
            if epoch == 5 and i == 0:
                client = TrainLocal(args, data_train, None, poison=1)
            elif i == 0:
                client = TrainLocal(args, data_train, None, poison=1)
            else:
                client = TrainLocal(args, data_train, dict_users[i], poison=0)
            w, _ = client.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals[i] = copy.deepcopy(w)

        # normal aggregating
        with torch.no_grad():
            for i in users:
                net_temp.load_state_dict(w_locals[i])
                if flag:
                    net_glob.load_state_dict(net_temp.state_dict())
                    for cv in net_glob.parameters():
                        cv.data *= 0.1
                    flag = False
                else:
                    for cv, ccv in zip(net_glob.parameters(), net_temp.parameters()):
                        cv.data += (ccv.data * 0.1).to(args.device)
        
        # with torch.no_grad():
        #     for i in users:
        #         net_temp.load_state_dict(w_locals[i])
        #         if i == 0 and epoch == 5:
        #             for name, data in net_temp.state_dict().items():
        #                 weight_accumulator[name].add_(data.data - net_glob.state_dict()[name].data).to('cuda')
        #         else:
        #             for name, data in net_temp.state_dict().items():
        #                 weight_accumulator[name].add_(data.data - net_glob.state_dict()[name].data).to('cuda')
        #
        #     for name, data in net_glob.state_dict().items():
        #         if data.data.dtype is not torch.int64:
        #             data.data.add_(weight_accumulator[name].data * 0.1)

        acc, loss = test(epoch, net_glob, data_train, args)
        acc_train.append(acc)
        loss_train.append(loss)
        acc, loss = test(epoch, net_glob, data_test, args)
        acc_test.append(acc)
        loss_test.append(loss)
        acc, _ = test(epoch, net_glob, data_green, args)
        acc_green.append(acc)
        acc, _ = test(epoch, net_glob, data_racing, args)
        acc_racing.append(acc)
        acc, _ = test(epoch, net_glob, data_vertical, args)
        acc_vertical.append(acc)

        if acc > best_acc:
            torch.save(net_glob.state_dict(), f'./save_model/cifar.pth')
            best_acc = acc

        plt.clf()

        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax1.set_xlabel('epoch')
        ax1.set_xlim(-0.5, 100.5)

        ax1.plot(x, loss_train, color='red', marker='*', label='loss_train')
        ax1.plot(x, loss_test, color='black', marker='+', label='loss_test')
        ax1.set_ylabel('loss')

        ax2 = ax1.twinx()
        ax2.plot(x, acc_train, color='orange', label='train')
        ax2.plot(x, acc_test, color='y', label='test')
        ax2.plot(x, acc_green, color='g', label='green')
        ax2.plot(x, acc_racing, color='c', label='racing')
        ax2.plot(x, acc_vertical, color='m', label='vertical')
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel('accuracy')
        fig.subplots_adjust(right=0.8)
        fig.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), borderaxespad=0.2,
                   bbox_transform=ax1.transAxes, frameon=False)

        plt.title('model training monitor')
        plt.pause(0.1)
        if epoch == 200:
            plt.ioff()
            plt.show()


if __name__ == '__main__':
    main()
