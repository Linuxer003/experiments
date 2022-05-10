import torch
from torchvision import transforms, datasets
from flbnn.net import ResNet18
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as f

model = ResNet18().to('cuda')
data = datasets.ImageFolder(root=f'/home/cifar10/test',
                            transform=transforms.Compose(
                                [
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4913, 0.4821, 0.4465], [0.2005, 0.1988, 0.2008])
                                ]
                            ))
model.load_state_dict(torch.load(f'/home/experiments/save_model/cifar.pth'))

loader = DataLoader(data, batch_size=100, shuffle=True)

total = 0
correct = 0
loss = 0.0
with torch.no_grad():
    for img, lab in loader:
        img, lab = img.to('cuda'), lab.to('cuda')
        total += lab.size(0)
        out = model(img)

        loss += f.cross_entropy(out, lab, reduction='sum').item()

        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(lab.data.view_as(pred)).long().cpu().sum()

print('accuracy: {:.3f}'.format(correct/total))
