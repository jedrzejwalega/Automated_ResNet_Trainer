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
from pickle import dumps, loads

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
        self.__results = namedtuple("results", "train_losses valid_losses train_accuracies valid_accuracies")
        self.__run = namedtuple("run", "valid_loss_mean model optimizer hyperparams epoch")
        self.best_run = self.__run(float("inf"), None, None, None, None)
    
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
    
    def train(self):
        all_hyperparameters = [v for v in self.hyperparameters.values()]
        hyperparam_combination = namedtuple("hyperparam_combination", "lr epoch_number batch_size gamma")
        averaged_epoch_results = namedtuple("averaged_epoch_results", "train_loss_mean valid_loss_mean train_accuracy_mean valid_accuracy_mean")
        for hyperparams in product(*all_hyperparameters):
                hyperparams = hyperparam_combination(*hyperparams)
                self.__reproducible(seed=42)
                self.__make_dataloaders(hyperparams.batch_size)
                self.__create_model(lr=hyperparams.lr)
                self.model = self.model.to(self.device)
                tb = self.__setup_tensorboard_basics(hyperparams)
                print(f"Starting training, lr={hyperparams.lr}, batch size={hyperparams.batch_size}, gamma={hyperparams.gamma} for {hyperparams.epoch_number} epochs")
                for epoch in range(hyperparams.epoch_number):
                    start = default_timer()
                    result = self.__train_valid_one_epoch(hyperparams.lr)
                    stop = default_timer()
                    mean_result = averaged_epoch_results(*map(mean, result))
                    self.__update_tensorboard_plots(tb, mean_result, epoch)
                    self.__save_model_if_best(mean_result, hyperparams, epoch+1)
                    print(f"Finished {epoch+1} epoch in {stop-start}s; train loss: {mean_result.train_loss_mean}, valid loss: {mean_result.valid_loss_mean}; train accuracy: {mean_result.train_accuracy_mean*100}%, valid_accuracy: {mean_result.valid_accuracy_mean*100}%")
                tb.close()
                print(f"Finished training of lr={hyperparams.lr}, batch size={hyperparams.batch_size}, gamma={hyperparams.gamma} for {hyperparams.epoch_number} epochs\n" + "-" * 20 + "\n")
    
    def __make_dataloaders(self, batch_size:int=64, num_workers:int=1, shuffle:bool=True, mode="train"):
        if mode=="train":
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=shuffle)
        elif mode=="test":
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    def __setup_tensorboard_basics(self, hyperparams, mode="train") -> SummaryWriter:
        tb = SummaryWriter(comment=f" {mode} lr={hyperparams.lr} epochs={hyperparams.epoch_number} batch size={hyperparams.batch_size}")
        images, labels = next(iter(self.train_loader))
        grid = torchvision.utils.make_grid(images)
        tb.add_image("images", grid)
        tb.add_graph(self.model, images.cuda())
        return tb    
    
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
        
        return self.__results(train_losses, valid_losses, train_accuracies, valid_accuracies)

    def __adjust_lr(self, new_lr:float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def __accuracy(self, pred:torch.Tensor, batch_y:torch.Tensor):
        predicted_classes = torch.argmax(pred, dim=1)
        correct = (predicted_classes == batch_y).float().sum()
        accuracy = (correct/batch_y.shape[0]).item()
        return accuracy
    
    def __update_tensorboard_plots(self, tb, mean_result, epoch):
        tb.add_scalar("Train_loss", mean_result.train_loss_mean, epoch)
        tb.add_scalar("Valid_loss", mean_result.valid_loss_mean, epoch)
        tb.add_scalar("Train_accuracy", mean_result.train_accuracy_mean, epoch)
        tb.add_scalar("Valid_accuracy", mean_result.valid_accuracy_mean, epoch)
        for param_name, param in self.model.named_parameters():
            tb.add_histogram(param_name, param, epoch)
            tb.add_histogram(f"{param_name} gradient", param.grad, epoch)
    
    def __save_model_if_best(self, mean_result, hyperparams, epoch):
        if mean_result.valid_loss_mean < self.best_run.valid_loss_mean:
            best_model = dumps(self.model)
            best_optimizer = dumps(self.optimizer)
            best_valid_loss_mean = mean_result.valid_loss_mean
            self.best_run = self.__run(best_valid_loss_mean, best_model, best_optimizer, hyperparams, epoch)

    def test(self):
        self.model = loads(self.best_run.model)
        self.__make_dataloaders(batch_size=self.best_run.hyperparams.batch_size,
                                shuffle=False,
                                mode="test")
        self.model.eval()
        word_end = "nd" if self.best_run.epoch >= 2 else "st"
        print(f"Starting testing, model from: {self.best_run.epoch}{word_end} epoch, lr={self.best_run.hyperparams.lr}, batch_size={self.best_run.hyperparams.batch_size}, gamma={self.best_run.hyperparams.gamma}")
        test_losses = []
        test_accuracies = []
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred = self.model(batch_x)
                test_accuracy = self.__accuracy(pred, batch_y)
                test_accuracies.append(test_accuracy)
                loss = self.loss_func(pred, batch_y)
                test_losses.append(loss.item())
        test_loss_mean = mean(test_losses)
        test_accuracy_mean = mean(test_accuracies)
        print(f"Finished testing, testing loss: {test_loss_mean}, test accuracy: {test_accuracy_mean}\n" + "-" * 20)

input_data.download_mnist("/home/jedrzej/Desktop/fmnist")
train_images, train_labels = input_data.load_mnist("/home/jedrzej/Desktop/fmnist")
test_images, test_labels = input_data.load_mnist("/home/jedrzej/Desktop/fmnist")
program = RunManager([0.001, 0.0001], [2])
program.model_params(10)
program.pass_datasets((train_images, train_labels), (test_images, test_labels))
program.train()
program.test()