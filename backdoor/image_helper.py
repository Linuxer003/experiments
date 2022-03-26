import os
import math
from shutil import copyfile

import torch
import random
import numpy as np
import torch.utils.data as data

from backdoor.net import MnistNet
from log import Logger


class Helper:
    def __init__(self, params, train_dataset, test_dataset):
        """

        :param params: training params, xxx.yaml

        """
        self.params = params
        self.device = torch.device(self.params['device'])

        self.best_loss = math.inf

        self.local_model = None
        self.target_model = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.logger = Logger(self.params['log_file']).logger
        try:
            os.mkdir(self.params['saved_model_name'])
        except FileExistsError:
            self.logger.info('Folder using to save model already exists!')

    @staticmethod
    def save_checkpoint(state, is_best, filename='checkpoint.pth'):
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth')

    def create_model(self):
        self.local_model = MnistNet().to(self.device)
        self.target_model = MnistNet().to(self.device)

    def poison_dataloader(self):
        """
        Poisoned dataset be produced, the poison image is empty.
        """
        indices = list()

        range_num_id = list(range(50000))
        for index in self.params['poison_images'] + self.params['poison_images_test']:
            if index in range_num_id:
                range_num_id.remove(index)

        # add random images to other parts of the batch
        for batches in range(0, self.params['size_of_secret_dataset']):
            range_iter = random.sample(range_num_id,
                                       self.params['batch_size'])
            indices.extend(range_iter)

        return data.DataLoader(self.train_dataset, batch_size=self.params['batch_size'],
                               sampler=data.sampler.SubsetRandomSampler(indices))

    def poison_test_dataset(self):
        return data.DataLoader(self.train_dataset, batch_size=self.params['batch_size'],
                               sampler=data.SubsetRandomSampler(range(1000)))

    def load_data(self, dataset, num_users):
        self.params.logger.logger.info('Loading data...')
        num_items = int(len(dataset) / num_users)
        dict_users, all_index = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_index, num_items, replace=False))
            all_index = list(set(all_index) - dict_users[i])
        return dict_users

    def get_train(self, indices):
        train_loader = data.DataLoader(self.train_dataset,
                                       batch_size=self.params['batch_size'],
                                       sampler=data.sampler.SubsetRandomSampler(indices))
        return train_loader

    def get_train_old(self, all_range, model_no):
        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       sub_indices))
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader

    def get_batch(self, bp_tt, evaluation=False):
        img, target = bp_tt
        img = img.cuda()
        target = target.cuda()
        if evaluation:
            img.requires_grad_(False)
            target.requires_grad_(False)
        return img, target
