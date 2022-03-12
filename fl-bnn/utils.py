import numpy as np


def iid(dataset, num_users):
    """
    sample IID client data from MNIST dataset.
    Args:
        dataset:
        num_users:

    Returns:

    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_index = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_index, num_items, replace=False))
        all_index = list(set(all_index - dict_users[i]))
    return dict_users
