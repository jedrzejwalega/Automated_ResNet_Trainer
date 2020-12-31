from fastai.vision.all import download_url
from pathlib import Path
import os
import numpy as np
import torch
import tarfile
import pickle
from PIL import Image
from torchvision import transforms



def download_cifar10(dir_name:str) -> None:
    cifar10_url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    dir_name = Path(dir_name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    download_url(cifar10_url, dir_name/"cifar10.tar.gz")
    if not os.path.isdir(dir_name/"cifar10"):
        tar = tarfile.open(dir_name/"cifar10.tar.gz", "r:gz")
        tar.extractall(path=dir_name)
        tar.close()
        os.rename(dir_name/"cifar-10-batches-py", dir_name/"cifar10")

def load_cifar10(dir_name:str, kind:str="train") -> None:
    dir_name = Path(dir_name)
    all_images = []
    all_labels = []
    for filename in os.listdir(dir_name/"cifar10"):
        if kind == "train" and "data" in filename:
            images, labels = unpack_data(dir_name, filename)
            all_images.append(images)
            all_labels.append(labels)
        elif kind == "test" and "test" in filename:
            images, labels = unpack_data(dir_name, filename)
            all_images.append(images)
            all_labels.append(labels)
    return torch.cat(all_images, dim=0), torch.cat(all_labels)

def unpack_data(dir_name: Path, filename: str):
    data = unpickle(dir_name/"cifar10"/filename)
    images = torch.from_numpy(data[b"data"]).float().reshape((-1, 3, 32, 32))
    labels = torch.tensor(data[b"labels"]).long()
    return (images, labels)

def unpickle(file: str):
    with open(file, 'rb') as handle:
        unpickled = pickle.load(handle, encoding='bytes')
    return unpickled