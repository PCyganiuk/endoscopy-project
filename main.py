import argparse
import os
import sys

from src.models import Models

DEFAULT_TEST_SIZE = 0.2
DEFAULT_TRAIN_SIZE = 0.8
EMPTY_FLOAT = -1

def dir_path(path):
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)

def file_path(path):
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)

def is_dir_empty(path):
    return not next(os.scandir(path), None)

def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Dataset preparator', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--train-size",
                        type=float,
                        default=0.8,
                        help="Size of train set split (sum of train+test+validation size must equal 1)",
                        required=False)
    parser.add_argument("--test-size",
                        type=float,
                        default=EMPTY_FLOAT,
                        help="Size of test set split (sum of train+test+validation size must equal 1)",
                        required=False)
    parser.add_argument("--output-path",
                        help="Output path for generated data (path content should be empty, no folders nor files inside, otherwise use -f to force clear)",
                        default="./data",
                        type=str,
                        required=False)
    parser.add_argument("--ers-path",
                        help="Path for ERS dataset (folder containing patients ids e.g. \"0001\")",
                        type=dir_path,
                        required=True)
    parser.add_argument("--galar-path",
                        help="Path for Galar dataset (folder containing patients ids e.g. \"0001\")",
                        type=dir_path,
                        required=False)
    parser.add_argument("--type-num",
                        help="Choose number from 0 to 4 to define which experiment to run",
                        type=int,
                        default=0,
                        required=True)
    parser.add_argument("--epochs",
                        help="Choose number of epochs",
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument("--k-folds",
                        help="choose number of k folds, default is 20",
                        type=int,
                        default=20,
                        required=False)
    parser.add_argument("--model-size",
                        help="Choose number from 0-2 where 0-small 1-medium 2-large",
                        type=int,
                        default=0,
                        required=True)
    parser.add_argument("--binary",
                        help="Choose 1 if binary(healthy/unhealthy), choose 0 if multilabel",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--verbose",
                        help="Choose number from 0-2",
                        type=int,
                        default=2,
                        required=True)
    parser.add_argument("--fisheye",
                        help="Choose number from 0-1",
                        type=int,
                        default=0,
                        required=True)
    return parser

def parse_args(args):
    parser = setup_argument_parser()
    args = parser.parse_args(args)

    if args.galar_path is None and args.ers_path is None:
        parser.error("At least one of --galar-path and --ers-path required")

    missing_sizes_count = 0
    for arg in [args.train_size, args.test_size]:
        if arg == EMPTY_FLOAT:
            missing_sizes_count += 1
    if missing_sizes_count == 2:
        args.train_size = DEFAULT_TRAIN_SIZE
        args.test_size = DEFAULT_TEST_SIZE
    elif missing_sizes_count > 1:
        if args.train_size == 1:
            args.test_size = 0
        elif args.test_size == 1:
            args.train_size = 0
        else:
            parser.error("Only one of --train-size and --test-size can be skipped (set one to 1.0 or provide another one)")
    if args.train_size == EMPTY_FLOAT:
        args.train_size = 1 - args.test_size
    if args.test_size == EMPTY_FLOAT:
        args.test_size = 1 - args.train_size
    if round(args.train_size + args.test_size) != 1.0:
        parser.error("Sum of --train-size and --test-size should be equal 1.0")
    
    return args

def main(args):
    setup_argument_parser()
    args = parse_args(args)
    Models(args).train()


if __name__ == '__main__':
    main(sys.argv[1:])