import numpy as np
import random
from collections import defaultdict


def iid(dataset, num_users, alpha=0.9):
    """
    sample IID client data from MNIST dataset.
    Args:
        dataset:
        num_users:
        alpha:
    Returns:

    """
    # n_classes = 10
    # label_distribution = np.random.dirichlet([0.9] * num_users, n_classes)

    num_items = int(len(dataset) / num_users)
    dict_users, all_index = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_index, num_items, replace=False))
        all_index = list(set(all_index) - dict_users[i])
    return dict_users


def sample_dirichlet(dataset, num_users, alpha=0.9):
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x

        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    class_size = len(cifar_classes[0])
    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(num_users * [alpha]))
        for user in range(num_users):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][
                           :min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][
                               min(len(cifar_classes[n]), no_imgs):]

    return per_participant_list
