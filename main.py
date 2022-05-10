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
    data_train = datasets.ImageFolder(root=r'/home/cifar10/train/', transform=tran_train)
    data_test = datasets.ImageFolder(root=r'/home/cifar10/test/', transform=tran_test)

    dict_users = iid(data_train, args.num_users)

    net_glob = ResNet18().to(args.device)
    net_temp = copy.deepcopy(net_glob).to(args.device)

    w_locals = [None for i in range(args.num_users)]
    users_num = [i for i in range(args.num_users)]
    loss_train = []
    acc_train = []
    acc_test = []

    plt.ion()
    x = []

    best_acc = 0.75

    for epoch in range(args.epochs):
        users = random.sample(users_num, 10)
        x.append(epoch)
        flag = True
        loss_locals = []

        if epoch == 30 or epoch == 50 or epoch == 80:
            args.lr *= 0.1

        for i in users:
            client = train_local(args, data_train, dict_users[i])
            w, loss = client.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

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

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        acc, _ = test(net_glob, data_train, args)
        acc_train.append(acc)
        acc, _ = test(net_glob, data_test, args)
        acc_test.append(acc)

        if acc > best_acc:
            torch.save(net_glob.state_dict(), f'./save_model/cifar.pth')
            best_acc = acc

        plt.clf()

        ax1 = plt.subplot(111)
        ax1.set_xlabel('epoch')

        ax1.plot(x, loss_train, color='red', label='loss')
        ax1.set_ylabel('loss')
        ax1.legend(loc=1)
        ax2 = ax1.twinx()
        ax2.plot(x, acc_train, color='blue', label='acc_train')
        ax2.plot(x, acc_test, color='green', label='acc_test')
        ax2.set_ylabel('acc')
        ax2.legend(loc=2)

        plt.title('model training monitor')
        plt.pause(0.1)
        if epoch == 499:
            plt.ioff()
            plt.show()


if __name__ == '__main__':
    main()
