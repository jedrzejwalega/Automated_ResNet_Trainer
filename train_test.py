import torch
import torch.nn as nn
import model
import torch.optim as optim
import input_data
import dataset
from typing import List,Tuple
from statistics import mean
from timeit import default_timer
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision
from functools import partial
from itertools import product
from collections import namedtuple
from pickle import dumps

class RunManager():
    def __init__(self, learning_rates:List[float], epochs:List[int], batch_size:List[int]=[64], gamma:List[float]=[0.1]):
        self.__reproducible(seed=42)
        self.hyperparameters = dict(learning_rates=learning_rates,
                                epochs=epochs,
                                batch_size=batch_size,
                                gamma=gamma)
        self.model = None
        self.optimizer = None
        self.train_data = None
        self.test_data = None
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.results = namedtuple("results", "train_losses valid_losses train_accuracies valid_accuracies")
        self.best_valid_loss = float("inf")
        self.best_model = None
        self.best_optimizer = None
    
    def __reproducible(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def model_params(self, out_activations:int, in_channels:int=1, optimizer=optim.SGD):
        self.__create_model = partial(self.__create_model, out_activations=out_activations, in_channels=in_channels, optimizer=optimizer)
    
    def __create_model(self, lr, out_activations:int, in_channels:int=1, optimizer=optim.SGD):
        self.model = model.ResNet50(out_activations, in_channels)
        self.optimizer = optimizer(self.model.parameters(), lr=lr, momentum=0.9)

    def pass_datasets(self, train_set:Tuple[torch.FloatTensor, torch.LongTensor], test_set:Tuple[torch.FloatTensor, torch.LongTensor]):
        train_set = dataset.ImageDataset(train_set)
        test_set = dataset.ImageDataset(test_set)
        train_len = len(train_set)
        lengths = [int(train_len*0.8), int(train_len*0.2)]
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(train_set, lengths=lengths, generator=torch.Generator().manual_seed(42))
        self.test_dataset = test_set
    
    def __make_dataloaders(self, batch_size:int=64, num_workers:int=1, shuffle:bool=True):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=shuffle)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self):
        parameters = [v for v in self.hyperparameters.values()]
        for lr, epoch_number, batch_size, gamma in product(*parameters):
                self.__reproducible(seed=42)
                self.__make_dataloaders(batch_size)
                self.__create_model(lr=lr)
                self.model = self.model.to(self.device)
                tb = SummaryWriter(comment=f" train lr={lr} epochs={epoch_number} batch size={batch_size}")
                self.__setup_tensorboard_basics(tb)
                print(f"Starting training, lr={lr}, batch size={batch_size}, gamma={gamma} for {epoch_number} epochs")
                for epoch in range(epoch_number):
                    start = default_timer()
                    result = self.__train_valid_one_epoch(lr)
                    stop = default_timer()
                    train_loss_mean = mean(result.train_losses)
                    valid_loss_mean = mean(result.valid_losses)
                    train_accuracy_mean = mean(result.train_accuracies)
                    valid_accuracy_mean = mean(result.valid_accuracies)
                    tb.add_scalar("Train_loss", train_loss_mean, epoch)
                    tb.add_scalar("Valid_loss", valid_loss_mean, epoch)
                    tb.add_scalar("Train_accuracy", train_accuracy_mean, epoch)
                    tb.add_scalar("Valid_accuracy", valid_accuracy_mean, epoch)
                    for param_name, param in self.model.named_parameters():
                        tb.add_histogram(param_name, param, epoch)
                        tb.add_histogram(f"{param_name} gradient", param.grad, epoch)
                    print(f"{self.best_valid_loss} vs {valid_loss_mean}")
                    if valid_loss_mean < self.best_valid_loss:
                        print("Weszlo")
                        self.best_model = dumps(self.model)
                        self.best_optimizer = dumps(self.optimizer)
                        self.best_valid_loss = valid_loss_mean
                    print(f"Finished {epoch+1} epoch in {stop-start}s; train loss: {train_loss_mean}, valid loss: {valid_loss_mean}; train accuracy: {train_accuracy_mean*100}%, valid_accuracy: {valid_accuracy_mean*100}%")
                tb.close()
                print(f"Finished training of lr={lr}, batch size={batch_size}, gamma={gamma} for {epoch_number} epochs\n" + "-" * 20 + "\n")
    
    def __train_valid_one_epoch(self, lr:float) -> Tuple[List[float], List[float]]:
        self.__adjust_lr(lr)
        self.model.train()
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred = self.model(batch_x)
            accuracy = self.__accuracy(pred, batch_y)
            train_accuracies.append(accuracy)
            loss = self.loss_func(pred, batch_y)
            train_losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()    
        with torch.no_grad():
            for batch_x, batch_y in self.valid_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred = self.model(batch_x)
                valid_accuracy = self.__accuracy(pred, batch_y)
                valid_accuracies.append(valid_accuracy)
                loss = self.loss_func(pred, batch_y)
                valid_losses.append(loss.item())
        
        return self.results(train_losses, valid_losses, train_accuracies, valid_accuracies)
    
    def __accuracy(self, pred:torch.Tensor, batch_y:torch.Tensor):
        predicted_classes = torch.argmax(pred, dim=1)
        correct = (predicted_classes == batch_y).float().sum()
        accuracy = (correct/batch_y.shape[0]).item()
        return accuracy
    
    def __setup_tensorboard_basics(self, tb) -> None:
        images, labels = next(iter(self.train_loader))
        grid = torchvision.utils.make_grid(images)
        tb.add_image("images", grid)
        tb.add_graph(self.model, images.cuda())

    def __adjust_lr(self, new_lr:float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
    
    def __add_tensorboard_stats(self, tb, loss, epoch, mode="train"):
        mode = mode.capitalize()
        tb.add_scalar(f"{mode}_loss", loss, epoch)
        if mode is "Train":
            for param_name, param in self.model.named_parameters():
                print(param)
                tb.add_histogram(f"{param_name} bias", self.model.param_name.bias, epoch)
                tb.add_histogram(f"{param_name} weights", self.model.param_name.weight, epoch)
                tb.add_histogram(f"{param_name} gradient", self.model.param_name.grad, epoch)

    def test(self):
        pass

input_data.download_mnist("/home/jedrzej/Desktop/fmnist")
train_images, train_labels = input_data.load_mnist("/home/jedrzej/Desktop/fmnist")
test_images, test_labels = input_data.load_mnist("/home/jedrzej/Desktop/fmnist")
program = RunManager([0.001, 0.0001], [2])
program.model_params(10)
program.pass_datasets((train_images, train_labels), (test_images, test_labels))
program.train()

print(type(program.best_model))