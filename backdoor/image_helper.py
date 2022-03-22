from helper import Helper
from net import MnistNet

import torch
from collections import defaultdict
import random
import numpy as np
import torch.utils.data as data
from torchvision import transforms, datasets


class ImageHelper(Helper):
    def __init__(self, args, name):
        super(ImageHelper, self).__init__(args, name)
        self.args = args
        self.train_dataset = None
        self.test_dataset = None
        pass

    def poison(self):
        return

    def create_model(self):
        local_model = MnistNet()
        local_model.to(self.args.device)
        target_model = MnistNet()
        target_model.to(self.args.device)

        self.local_model = local_model
        self.target_model = target_model

    def poison_dataset(self):
        """
        Poisoned dataset be produced, the poison image is empty.
        """
        classes = {}
        for index, x in enumerate(self.train_dataset):
            _, label = x
            if index in self.args['poison_images'] or index in self.args['poison_images_test']:
                continue
            if label in classes:
                classes[label].append(index)
            else:
                classes[label] = [index]

        indices = list()

        range_no_id = list(range(50000))
        for image in self.args['poison_images'] + self.args['poison_images_test']:
            if image in range_no_id:
                range_no_id.remove(image)

        # add random images to other parts of the batch
        for batches in range(0, self.args['size_of_secret_dataset']):
            range_iter = random.sample(range_no_id,
                                       self.args['batch_size'])
            indices.extend(range_iter)

        return data.DataLoader(self.train_dataset, batch_size=self.args['batch_size'],
                               sampler=data.sampler.SubsetRandomSampler(indices))

    def poison_test_dataset(self):
        return data.DataLoader(self.train_dataset, batch_size=self.args['batch_size'],
                               sampler=data.SubsetRandomSampler(range(1000)))

    def load_data(self, dataset, num_users):
        self.args.logger.logger.info('Loading data...')
        num_items = int(len(dataset) / num_users)
        dict_users, all_index = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_index, num_items, replace=False))
            all_index = list(set(all_index) - dict_users[i])
        return dict_users

    def get_train(self, indices):
        train_loader = data.DataLoader(self.train_dataset,
                                       batch_size=self.args['batch_size'],
                                       sampler=data.sampler.SubsetRandomSampler(indices))
        return train_loader

    def get_train_old(self, all_range, model_no):
        data_len = int(len(self.train_dataset) / self.args['number_of_total_participants'])
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
