import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated learning
    parser.add_argument('--epochs', type=int, default=100, help='number of the train epochs')
    parser.add_argument('--num_users', type=int, default=100, help='number of client chosen to upload the weight')

    args = parser.parse_args()
    return args
