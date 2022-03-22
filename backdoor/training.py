import random
import sys
sys.path.append('/Users/wgh/PycharmProjects/experiments/')
import log
import yaml
import torch
import argparse

from image_helper import ImageHelper


def train():

    pass


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.logger = log.Logger(f'log.txt')

    args.logger.logger.info(r'Loading parameters...')
    # params_load = None
    with open(r'backdoor/params.yaml', 'r') as f:
        params_load = yaml.safe_load(f)
    args.params_load = params_load

    helper = ImageHelper(args, f'mnist_net')
    dataset = None

    dict_users = helper.load_data(dataset, params_load['num_users'])
    helper.create_model()

    adversary_list = random.sample(range(params_load['num_clients']), params_load['num_adversaries'])
    args.logger.logger.info(f'Poisoned following participants: {adversary_list}')

    best_loss = float('inf')
