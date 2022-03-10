

def iid(dataset, num_users):
    """
    sample IID client data from MNIST dataset.
    Args:
        dataset:
        num_users:

    Returns:

    """
    num_client = int(len(dataset) / num_users)
