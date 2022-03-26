import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated learning
    parser.add_argument('--epochs', type=int, default=10, help='number of the train epochs')
    parser.add_argument('--num_users', type=int, default=100, help='number of client chosen to upload the weight')
    parser.add_argument('--local_bs', type=int, default=10, help='')
    parser.add_argument('--local_ep', type=int, default=5, help='')
    parser.add_argument('--test_bs', type=int, default=100, help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--momentum', type=float, default=0.5, help='')

    args = parser.parse_args()
    return args
