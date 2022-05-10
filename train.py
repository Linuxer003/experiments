from torchvision import transforms, datasets
from flbnn.net import *
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt


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
load_train = DataLoader(data_train, batch_size=256, shuffle=True)
load_test = DataLoader(data_test, batch_size=500, shuffle=True)

model = ResNet18().to('cuda')

optim = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

losses = []
accuracy = []
epoch = []
plt.ion()

for i in range(1, 101):
    epoch.append(i)
    if i == 51:
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    sum_loss = 0.0

    for inputs, labels in tqdm(load_train, desc=f'Epoch {i}: '):

        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        sum_loss += loss.item()

    losses.append(sum_loss / len(data_train))

    model.eval()

    correct, total = 0, 0
    for inputs, labels in load_test:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        _, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()

    accuracy.append(100. * correct / total)

    plt.clf()
    fig, ax1 = plt.subplots()
    plt.xlabel('epoch')

    ax1.plot(epoch, losses, color='red', label='loss')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.plot(epoch, accuracy, color='green', label='accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_ylim(-0.5, 100.5)

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.title('model training monitor')
    plt.xlim(-0.5, 100.5)

    plt.pause(0.1)
    if epoch == 99:
        plt.ioff()
        plt.show()
