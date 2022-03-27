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
    poison_data = datasets.ImageFolder(root=r'D:/data/MNIST/poison/', transform=trans)
    dict_users = iid(data_train, params['num_users'])

    # Declare the model to train.
    net_glob = MnistNet().to(torch.device(params['device']))

    net_temp = MnistNet()
    w_locals = [None for i in range(params['num_users'])]

    loss_train = []

    plt.ion()
    x = []

    for epoch in range(params['epochs']):
        x.append(epoch)
        flag = True
        loss_locals = []
        for i in range(params['num_users']):
            if i < 10 and epoch in range(30, 50):
                client = train_local(params, data_train, dict_users[i], poison_data, poison=True)
            else:
                client = train_local(params, data_train, dict_users[i], poison_data, poison=False)
            w, loss = client.train(net=copy.deepcopy(net_glob).to(torch.device(params['device'])))
            w_locals[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

        with torch.no_grad():
            for w in w_locals:
                net_temp.load_state_dict(w)
                net_temp.pre_com()
                if flag:
                    net_glob.load_state_dict(net_temp.state_dict())
                    for cv in net_glob.parameters():
                        cv.data *= 0.01
                    flag = False
                else:
                    for cv, ccv in zip(net_glob.parameters(), net_temp.parameters()):
                        cv.data += (ccv.data * 0.01).to(torch.device(params['device']))

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Epoch {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        plt.clf()
        plt.plot(x, loss_train)
        plt.pause(0.1)
        if epoch == 9:
            plt.ioff()
            plt.show()

    net_glob.eval()
    acc_train, loss_train = test(net_glob, data_train, params)
    acc_test, loss_test = test(net_glob, data_test, params)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


