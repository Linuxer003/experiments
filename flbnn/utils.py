import numpy as np


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
