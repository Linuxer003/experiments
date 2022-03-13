from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt

from options import *
from utils import *
from net import *
from train_local import *
from test_glob import *


def main():
    args = args_parser()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_train = datasets.MNIST(root=r'D:/data/', train=True, download=False, transform=trans)
    data_test = datasets.MNIST(root=r'D:/data/', train=False, download=False, transform=trans)
    dict_users = iid(data_train, args.num_users)

    net_glob = MnistNet().to(args.device)

    net_temp = MnistNet()
    w_locals = [None for i in range(args.num_users)]

    loss_train = []

    plt.ion()
    x = []

    for epoch in range(args.epochs):
        x.append(epoch)
        flag = True
        loss_locals = []
        for i in range(args.num_users):
            client = train_local(args, data_train, dict_users[i])
            w, loss = client.train(net=copy.deepcopy(net_glob).to(args.device))
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
                        cv.data += (ccv.data * 0.01).to(args.device)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Epoch {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        plt.clf()
        plt.plot(x, loss_train)
        plt.pause(0.1)
        if epoch == 99:
            plt.ioff()
            plt.show()

    net_glob.eval()
    acc_train, loss_train = test(net_glob, data_train, args)
    acc_test, loss_test = test(net_glob, data_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


if __name__ == '__main__':
    main()