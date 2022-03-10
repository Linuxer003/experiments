from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

trans = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
dataset = datasets.ImageFolder(root=r'D:/data/tiny-imagenet-200/train/', transform=trans)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

mean = torch.zeros(3)
std = torch.zeros(3)

for data, _ in tqdm(loader):
    for i in range(3):
        mean[i] += data[:, i, :, :].mean()
        std[i] += data[:, i, :, :].std()
mean.div_(len(loader))
std.div_(len(loader))
print(mean, std)
