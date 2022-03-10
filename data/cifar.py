from torchvision import datasets
from utils import utils
from matplotlib import image


def get_cifar10_data(data_param, transform):
    """

    Args:
        data_param
        transform:

    Returns:

    """
    data_path = utils.config_read('input_1', data_param)
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return dataset


def cifar10conv2jpg():
    """

    Returns:

    """
    for i in range(1, 6):
        data, label = utils.unpickle(r'./input_1/cifar-10-batches-py/data_batch_'+str(i))
        for j in range(10000):
            img = data[j]
            img = img.reshape(3, 32, 32)
            img = img.transpose(1, 2, 0)
            img_name = r'./input_1/cifar10/train/'+str(label[j])+'/batch_'+str(i)+'_'+str(j)+'.jpg'
            image.imsave(img_name, img)
    data, label = utils.unpickle(r'./input_1/cifar-10-batches-py/test_batch')
    for i in range(10000):
        img = data[i]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img_name = r'./input_1/cifar10/test/'+str(label[i])+'/'+str(i)+'.jpg'
        image.imsave(img_name, img)
