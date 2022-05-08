from torchvision import datasets

data = datasets.ImageFolder(root=f'/opt/data/common/ImageNet/ILSVRC2012/train/')

for _, x in enumerate(data):
    _, label = x
    print(label)
    break

