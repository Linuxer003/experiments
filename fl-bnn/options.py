import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated learning
    parser.add_argument('--epochs', type=int, default=100, help='')
