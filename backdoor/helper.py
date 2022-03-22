import math
import os
from shutil import copyfile

import torch


class Helper:

    def __init__(self, args, name):

        self.target_model = None
        self.local_model = None

        self.start_epoch = None
        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.args = args
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'./saved_model/model_{self.name}'

        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            self.args.logger.logger.info('Folder already exists')

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        if not self.args['saved_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)
