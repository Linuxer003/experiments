import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class DatasetSplit(Dataset):
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = list(indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        img, label = self.dataset[self.indexes[item]]
        return img, label


class train_local:
    def __init__(self, params, dataset, indexes, poison_data, poison):
        self.params = params
        self.loss_func = nn.CrossEntropyLoss()
        self.data_loader = DataLoader(DatasetSplit(dataset, indexes), batch_size=self.params['local_bs'], shuffle=True)
        self.poison_loader = DataLoader(poison_data, batch_size=45, shuffle=True)
        self.poison = poison

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.params['lr'], momentum=self.params['momentum'])

        epoch_loss = []
        for i in range(self.params['local_ep']):
            batch_loss = []
            for img, lab in self.data_loader:
                img, lab = img.to(torch.device(self.params['device'])), lab.to(torch.device(self.params['device']))
                optimizer.zero_grad()
                out = net(img)
                loss = self.loss_func(out, lab)

                loss.backward()
                for p in net.parameters():
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)
                optimizer.step()
                for p in net.parameters():
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp(-1, 1))
                batch_loss.append(loss.item())

            # training on poison data
            if self.poison:
                for img, lab in self.poison_loader:
                    img, lab = img.to(torch.device(self.params['device'])), lab.to(torch.device(self.params['device']))
                    optimizer.zero_grad()
                    out = net(img)
                    loss = self.loss_func(out, lab)

                    loss.backward()
                    for p in net.parameters():
                        if hasattr(p, 'org'):
                            p.data.copy_(p.org)
                    optimizer.step()
                    for p in net.parameters():
                        if hasattr(p, 'org'):
                            p.org.copy_(p.data.clamp(-1, 1))
                    batch_loss.append(loss.item())
                    break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
