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

    def load_data(self):
        self.params.logger.logger.info('Loading data...')
        num_items = int(len(self.train_dataset) / self.params['num_users'])
        dict_users, all_index = {}, [i for i in range(len(self.train_dataset))]
        for i in range(self.params['num_users']):
            dict_users[i] = set(np.random.choice(all_index, num_items, replace=False))
            all_index = list(set(all_index) - dict_users[i])
        return dict_users
