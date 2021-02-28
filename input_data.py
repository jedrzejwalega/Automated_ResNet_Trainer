from fastai.vision.all import download_url
from pathlib import Path
import os
import numpy as np
import gzip
import torch
import argparse


# Parse input arguments
def get_user_input() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", 
                        type=str,
                        help="Path to download data to. Only downloads if the data is missing, otherwise uses existing one.", 
                        required=True)
    parser.add_argument("--learning_rates", "-lr", 
                        nargs="+",
                        type=float,
                        help="An arbitrary number of learning rates to use in model training",
                        required=False,
                        default=[])
    parser.add_argument("--gamma", "-g", 
                        nargs="+",
                        type=float,
                        help="An arbitrary number of gammas (learning rate decay) to use in model training. Defaults to 0.1. Add 1 for no gamma at all.",
                        required=False,
                        default=[0.1])
    parser.add_argument("--batch_size", "-bs", 
                        nargs="+",
                        type=int,
                        help="An arbitrary number of batch sizes to use in model training. Defaults to 64.",
                        required=False,
                        default=[128])
    parser.add_argument("--epochs", "-e", 
                        nargs="+",
                        type=int,
                        help="An arbitrary number of epoch limits to train for for each hyperparameter combination.",
                        required=True)
    parser.add_argument("--shuffle", "-s", 
                        nargs="+",
                        type=bool,
                        help="Add one of three options: True, False or True False. Specifies whether the dataset will be shuffled during training. Defaults to True.",
                        required=False,
                        default=[True])
    parser.add_argument("--find_lr", "-f",
                        help="True/False - whether to use fastai's learning rate finder to find optimal learning rate. If True, you shouldn't pass your own learning rates.",
                        action="store_true")
    parser.add_argument("--test", "-t",
                        help="Whether to automaticaly test best model from the training sessions on downloaded test set. Defaults to False.",
                        action="store_true")
    parser.add_argument("--gamma_step", "-gs", 
                        nargs="+",
                        type=int,
                        help="An arbitrary number of gamma steps (how many epochs till gamma is applied to change the learning rate). Defaults to 3.",
                        required=False,
                        default=[3])
    parser.add_argument("--weight_decay", "-wd", 
                        nargs="+",
                        type=float,
                        help="An arbitrary number of weight decay values to apply to the optimizer.",
                        required=False,
                        default=[1e-4])
    parser.add_argument("--momentum", "-m", 
                    nargs="+",
                    type=float,
                    help="An arbitrary number of momentum values to apply to the optimizer.",
                    required=False,
                    default=[0.9])
    parser.add_argument("--write_model", "-w",
                        type=str,
                        help="Save best model from the training sessions (based od validation accuracy) into a given path.",
                        required=False)
    parser.add_argument("--architecture", "-a", 
                        nargs="+",
                        type=str,
                        help="An arbitrary number of architectures to use in model training. Available nets: resnet18, resnet34, resnet50, resnet101, resnet152",
                        required=True)
    parser.add_argument("--find_gamma_step", "-fg", 
                        help="Whether to find a gamma step based on loss plot slope.",
                        action="store_true")

    args = parser.parse_args()
    args.architecture = list(map(str.lower, args.architecture))
    available_nets = set(["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    assert all(net in available_nets for net in args.architecture), "Given model architecture is not available"
    if args.find_lr:
        assert not args.learning_rates, "You cannot pass custom learning rates when using automatic best learning rate option"

    return args