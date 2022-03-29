import torch
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt
import yaml

import log
from flbnn.utils import *
from flbnn.net import *
from flbnn.train_local import *
from flbnn.test_glob import *


if __name__ == '__main__':
    with open(r'./flbnn/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    logger = log.Logger(r'./log.txt').logger
    logger.info(r'Staring experiments...')

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_train = datasets.MNIST(root=r'D:/data/', train=True, download=False, transform=trans)
    data_test = datasets.MNIST(root=r'D:/data/', train=False, download=False, transform=trans)
    trans_img = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_poison = datasets.ImageFolder(root=r'D:/data/MNIST/poison/', transform=trans_img)
    data_poison_test = datasets.ImageFolder(root=r'D:/data/MNIST/poison_test/', transform=trans_img)
    dict_users = iid(data_train, params['num_users'])

    # Declare the model to train.
    net_glob = MnistNet().to(torch.device(params['device']))

    net_temp = MnistNet().to(torch.device(params['device']))
    w_locals = [None for i in range(params['num_users'])]

    loss_train = []
    asr = []
    plt.ion()
    x = []

    flag = True
    for epoch in range(params['epochs']):
        net_glob.train()
        x.append(epoch)
        loss_locals = []
        for i in range(params['num_users']):
            if i < 10 and epoch in range(50, 80):
                client = train_local(params, data_train, dict_users[i], data_poison, poison=True)
            else:
                client = train_local(params, data_train, dict_users[i], data_poison, poison=False)
            w, loss = client.train(net=copy.deepcopy(net_glob).to(torch.device(params['device'])))
            w_locals[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
        t = 0
        net_glob.eval()
        with torch.no_grad():
            for w in w_locals:
                net_temp.load_state_dict(w)
                net_temp.pre_com()
                if t < 10 and epoch in range(50, 80):

                    for cv, ccv in zip(net_glob.parameters(), net_temp.parameters()):
                        cv.data += ccv.data * 0.01

                else:
                    if flag:
                        net_glob.load_state_dict(net_temp.state_dict())
                        for cv in net_glob.parameters():
                            cv.data *= 0.01
                        flag = False
                    else:
                        for cv, ccv in zip(net_glob.parameters(), net_temp.parameters()):
                            cv.data += ccv.data * 0.01
                t += 1

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        acc_test, loss_test = test(net_glob, data_test, params)
        acc_attack, _ = test(net_glob, data_poison_test, params)
        asr.append(acc_attack)

        logger.info('Epoch {:3d}, model loss {:.3f}, test acc {:.3f}, test loss {:.3f}, attack success rate {:.3f}'
                    .format(epoch, loss_avg, acc_test, loss_test, acc_attack))

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(x, loss_train)
        plt.subplot(1, 2, 2)
        plt.plot(x, asr)
        plt.pause(0.1)
        if epoch == 99:
            plt.ioff()
            plt.show()

    net_glob.eval()
    acc_train, loss_train = test(net_glob, data_train, params)
    acc_test, loss_test = test(net_glob, data_test, params)
    acc_poison, loss_poison = test(net_glob, data_poison_test, params)
    logger.info('Model eventually Test acc {:.3f} loss {:.3f}, poison asr {:.3f} loss {:.3f}'
                .format(acc_test, loss_test, acc_poison, loss_poison))
    torch.save(net_glob.state_dict(), r'./save_model/model_attack.pth')


