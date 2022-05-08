import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
    def __init__(self, args, dataset, indexes):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.data_loader = DataLoader(DatasetSplit(dataset, indexes), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for i in range(self.args.local_ep):
            batch_loss = []
            for img, lab in tqdm(self.data_loader):
                img, lab = img.to(self.args.device), lab.to(self.args.device)
                optimizer.zero_grad()
                out = net(img)
                loss = self.loss_func(out, lab)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
