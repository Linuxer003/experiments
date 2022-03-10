import os
import pandas as pd
import numpy as np
import PIL
import utils
from torchvision import datasets
import torch.utils as tutils


def get_gtsrb_data(transform):
    """

    Args:
        transform:

    Returns:

    """
    data_path = utils.config_read('input_1', 'gtsrb')
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_size = round(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = tutils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def gtsrb_conv_jpg():
    train_root = r'./input_1/gtsrb/train'
    file_dir = r'D:/download/GTSRB/Final_Training/Images'
    directories = [file for file in os.listdir(file_dir)
                   if os.path.isdir(os.path.join(file_dir, file))]
    for files in directories:
        path = os.path.join(train_root, files)
        if not os.path.exists(path):
            os.makedirs(path)
        data_dir = os.path.join(file_dir, files)

        for f in os.listdir(data_dir):
            if f.endswith('.csv'):
                csv_dir = os.path.join(data_dir, f)
        csv_data = pd.read_csv(csv_dir)
        csv_data_array = np.array(csv_data)

        for i in range(csv_data_array.shape[0]):
            csv_data_list = csv_data_array[i].tolist()[0].split(';')
            sample_dir = os.path.join(data_dir, csv_data_list[0])
            img = PIL.Image.open(sample_dir)
            box = (int(csv_data_list[3]), int(csv_data_list[4]), int(csv_data_list[5]), int(csv_data_list[6]))
            roi_img = img.crop(box)

            new_dir = os.path.join(path, csv_data_list[0].split('.')[0]+'.jpg')
            roi_img.save(new_dir, 'JPEG')
            img.close()
        csv_data.close()
