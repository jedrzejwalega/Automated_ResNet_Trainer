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
import cifar10_dataset
import dataset

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
    all_images = np.vstack(all_images).reshape(-1, 3, 32, 32)
    return torch.from_numpy(all_images), torch.cat(all_labels)

def unpack_data(dir_name: Path, filename: str) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    data = unpickle(dir_name/"cifar10"/filename)
    images = torch.from_numpy(data["data"]).float()
    labels = torch.tensor(data["labels"]).long()
    return (images, labels)

def unpickle(file: str):
    with open(file, 'rb') as handle:
        unpickled = pickle.load(handle, encoding='latin1')
    return unpickled

def augment_data(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    transformsy = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,])
    
    image = transformsy(image)
    return image
# train_images, train_labels = cifar10_dataset.load_cifar10("/home/jedrzej/Desktop/")
# test_images, test_labels = cifar10_dataset.load_cifar10("/home/jedrzej/Desktop/", kind="test")
# train_set = dataset.ImageDataset((train_images, train_labels), transform=transformsy)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

# for x, in train_loader:
#     print(x.shape)