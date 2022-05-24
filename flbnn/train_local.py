import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from flbnn.test_glob import *
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = list(indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        img, label = self.dataset[self.indexes[item]]
        return img, label


class TrainLocal:
    def __init__(self, args, dataset, indexes, poison=0):
        self.poison = poison
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.test_dataset = datasets.ImageFolder(root=r'/home/data/cifar10/test/', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4913, 0.4821, 0.4465], [0.2005, 0.1988, 0.2008])
        ]))
        if poison == 2:
            self.data_loader = DataLoader(
                datasets.ImageFolder(
                    root=r'/home/data/client_0_backdoor/',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4913, 0.4821, 0.4465], [0.2005, 0.1988, 0.2008])
                    ])
                ),
                batch_size=50,
                shuffle=True
            )
        elif poison == 1:
            self.data_loader = DataLoader(
                datasets.ImageFolder(
                    root=r'/home/data/client_0/',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4913, 0.4821, 0.4465], [0.2005, 0.1988, 0.2008])
                    ])
                ),
                batch_size=50,
                shuffle=True
            )
        else:
            self.data_loader = DataLoader(DatasetSplit(dataset, indexes), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net_temp = copy.deepcopy(net)
        net.train()
        net.to('cuda')
        if self.poison == 2:
            self.args.local_ep = 30
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.5)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
            scheduler = None
        epoch_loss = []
        for i in range(self.args.local_ep):
            batch_loss = []
            for img, lab in self.data_loader:
                img, lab = img.to(self.args.device), lab.to(self.args.device)

                out = net(img)
                optimizer.zero_grad()
                loss = self.loss_func(out, lab)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if self.poison == 2:
                scheduler.step()
                test('client_epoch', net, self.test_dataset, self.args)

        if self.poison == 2:
            for cv, ccv in zip(net.parameters(), net_temp.parameters()):
                cv.data.copy_(cv.data * 10 - ccv.data * 9)

            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
