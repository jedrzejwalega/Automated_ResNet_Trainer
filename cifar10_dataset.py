from fastai.vision.all import download_url
from pathlib import Path
import os
import numpy as np
import torch
import tarfile
import pickle
from PIL import Image
from torchvision import transforms
from typing import Tuple


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
    train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]
        
    dir_name = Path(dir_name)
    all_images = []
    all_labels = []
    for file_name, checksum in train_list:
        file_path = os.path.join(dir_name/"cifar10", file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            all_images.append(entry['data'])
            if 'labels' in entry:
                all_labels.extend(entry['labels'])
            else:
                all_labels.extend(entry['fine_labels'])
    all_images = np.vstack(all_images).reshape(-1, 3, 32, 32)
    return torch.from_numpy(all_images), torch.tensor(all_labels)

def unpack_data(dir_name: Path, filename: str) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    data = unpickle(dir_name/"cifar10"/filename)
    images = torch.from_numpy(data["data"])
    labels = torch.tensor(data["labels"]).long()
    return (images, labels)


def unpickle(file: str):
    with open(file, 'rb') as handle:
        unpickled = pickle.load(handle, encoding='latin1')
    return unpickled

def augment_data_train(image:torch.FloatTensor) -> torch.FloatTensor:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transforms = transforms.Compose([transforms.ToPILImage(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize,])
    image = train_transforms(image)
    return image

def augment_data_valid(image):
    image = image.float()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_transforms = transforms.Compose([normalize,])
    image = valid_transforms(image)
    return image