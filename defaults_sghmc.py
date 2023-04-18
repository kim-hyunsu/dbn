import argparse


PIXEL_MEAN = (0.49, 0.48, 0.44)
PIXEL_STD = (0.2, 0.2, 0.2)


def default_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='./data/', type=str,
                        help='root directory containing datasets (default: ./data/)')
    parser.add_argument('--data_name', default='CIFAR100_x32', type=str,
                        choices=['CIFAR10_x32', 'CIFAR100_x32',])
    parser.add_argument('--data_augmentation', default='none', type=str,
                        choices=['none',])
    parser.add_argument('--data_proportional', default=1.0, type=float,
                        help='use the proportional train split if specified (default: 1.0)')

    parser.add_argument('--model_depth', default=20, type=int,
                        choices=[20,])
    parser.add_argument('--model_width', default=1, type=int,
                        help='widen factor (default: 1)')
    parser.add_argument('--model_style', default='FRN-Swish', type=str,
                        choices=['FRN-Swish', "BN-ReLU"])  # FilterResponseNorm / Swish activation

    parser.add_argument('--optim_bs', default=80, type=int,
                        help='mini-batch size (default: 80)')

    return parser
