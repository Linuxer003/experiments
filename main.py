import copy
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import log
from flbnn.options import *
from flbnn.utils import *
from flbnn.net import *
from flbnn.train_local import *
from flbnn.test_glob import *


def main():
    args = args_parser()
    args.logger = log.Logger(r'log.txt').logger
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_train = datasets.ImageFolder(root=r'/home/experiments/data/mnist/train/', transform=trans)
    data_test = datasets.ImageFolder(root=r'/home/experiments/data/mnist/test/', transform=trans)

    dict_users = iid(data_train, args.num_users)

    net_glob = CNNMnist().to(args.device)

    net_temp = CNNMnist()
    w_locals = [None for i in range(args.num_users)]

    loss_train = []
    acc_train = []
    acc_test = []

    plt.ion()
    x = []

    for epoch in range(args.epochs):
        x.append(epoch)
        flag = True
        loss_locals = []
        if epoch == 30 or epoch == 50 or epoch == 80:
            args.lr *= 0.5
        for i in range(args.num_users):
            client = train_local(args, data_train, dict_users[i])
            w, loss = client.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

        with torch.no_grad():
            for w in w_locals:
                net_temp.load_state_dict(w)
                if flag:
                    net_glob.load_state_dict(net_temp.state_dict())
                    for cv in net_glob.parameters():
                        cv.data *= 0.01
                    flag = False
                else:
                    for cv, ccv in zip(net_glob.parameters(), net_temp.parameters()):
                        cv.data += (ccv.data * 0.01).to(args.device)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        acc, _ = test(net_glob, data_train, args)
        acc_train.append(acc)
        acc, _ = test(net_glob, data_test, args)
        acc_test.append(acc)

        plt.clf()
        fig, ax1 = plt.subplots()
        plt.xlabel('epoch')

        ax1.plot(x, loss_train, color='red', label='loss')
        ax1.set_ylabel('loss')
        plt.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(x, acc_train, color='blue', label='acc_train')
        ax2.plot(x, acc_test, color='green', label='acc_test')
        ax2.set_ylabel('acc')
        plt.legend(loc=1)

        plt.title('model training monitor')
        plt.pause(0.1)
        if epoch == 99:
            plt.ioff()
            plt.show()

    dataframe = pd.DataFrame({
        'loss': loss_train,
        'acc_train': acc_train,
        'acc_test': acc_test
    })
    dataframe.to_csv('float.csv', sep=',')
    net_glob.eval()
    acc_train, loss_train = test(net_glob, data_train, args)
    acc_test, loss_test = test(net_glob, data_test, args)
    args.logger.info("Training accuracy: {:.2f}".format(acc_train))
    args.logger.info("Testing accuracy: {:.2f}".format(acc_test))
    torch.save(net_glob.state_dict(), './save_model/float.pth')


if __name__ == '__main__':
    main()
