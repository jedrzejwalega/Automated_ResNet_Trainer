import torch
import torch.nn as nn
import model
import torch.optim as optim
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
import fastai.learner
import fastai.data
from sys import float_info


class RunManager():
    def __init__(self, 
                 learning_rates:List[float], 
                 epochs:List[int], architectures:List[str], 
                 batch_size:List[int]=[64], 
                 gamma:List[float]=[0.1], 
                 shuffle:List[bool]=[True], 
                 optimizer=optim.SGD, 
                 find_lr=False, gamma_step=[1], 
                 find_gamma_step=False,
                 transform=None):

        self.reproducible(seed=42)
        if find_lr:
            learning_rates = [None]
        self.find_gamma_step = find_gamma_step
        if self.find_gamma_step:
            gamma_step = ["AUTO"]
        self.hyperparameters = dict(learning_rates=learning_rates,
                                epochs=epochs,
                                batch_size=batch_size,
                                gamma=gamma,
                                shuffle=shuffle,
                                gamma_step=gamma_step,
                                architectures=architectures)
        self.transform = transform
        self.model = None
        self.optimizer_algorythm = optimizer
        self.optimizer = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.results = namedtuple("results", "train_losses valid_losses train_accuracies valid_accuracies train_batch_times valid_batch_times")
        self.run = namedtuple("run", "valid_loss_mean model optimizer hyperparams epoch")
        self.best_run = self.run(float("inf"), None, None, None, None)
    
    def reproducible(self, seed:int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def model_params(self, out_activations:int, in_channels:int=1):
        self.create_model = partial(self.create_model, out_activations=out_activations, in_channels=in_channels)
    
    def create_model(self, architecture, out_activations:int, in_channels:int=1):
        available_architectures = {"resnet18":model.ResNet18,
                                   "resnet34":model.ResNet34,
                                   "resnet50":model.ResNet50,
                                   "resnet101":model.ResNet101,
                                   "resnet152":model.ResNet152}
        chosen_model = available_architectures[architecture]
        self.model = chosen_model(out_activations, in_channels)

    def pass_datasets(self, train_set:Tuple[torch.FloatTensor, torch.LongTensor], test_set:Tuple[torch.FloatTensor, torch.LongTensor]):
        assert len(train_set[0].shape) > 3 and len(test_set[0].shape) > 3, "You have to provide data in the form of an at least rank 4 tensor, with last 3 dimensions being: channels, height, width"
        channels, out_activations = self.get_model_params(train_set)
        self.model_params(out_activations=out_activations, in_channels=channels)
        train_set = dataset.ImageDataset(train_set, transform=self.transform)
        test_set = dataset.ImageDataset(test_set, transform=self.transform)
        train_len = len(train_set)
        lengths = [int(train_len*0.8), int(train_len*0.2)]
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(train_set, lengths=lengths, generator=torch.Generator().manual_seed(42))
        self.test_dataset = test_set
    
    def get_model_params(self, train_set):
        in_channels = train_set[0].shape[-3]
        out_activations = len(torch.unique(train_set[1]))
        return in_channels, out_activations
    
    def train(self):
        if self.hyperparameters["learning_rates"] == [None]:
            best_learning_rates = self.best_lr_for_hyperparameters()
        all_hyperparameters = [v for v in self.hyperparameters.values()]
        hyperparam_combination = namedtuple("hyperparam_combination", "lr epoch_number batch_size gamma shuffle gamma_step architecture")
        averaged_epoch_results = namedtuple("averaged_epoch_results", "train_loss_mean valid_loss_mean train_accuracy_mean valid_accuracy_mean train_batch_time_mean valid_batch_time_mean")
        slope_tracker = namedtuple("slope_tracker", "epoch previous_slope new_slope")

        for hyperparams in product(*all_hyperparameters):
            hyperparams = hyperparam_combination(*hyperparams)
            if not hyperparams.lr:
                fitting_best_lr = best_learning_rates[(hyperparams.batch_size, hyperparams.shuffle)]
                hyperparams = hyperparam_combination(fitting_best_lr, *hyperparams[1:])
            self.reproducible(seed=42)
            self.make_dataloaders(hyperparams.batch_size, shuffle=hyperparams.shuffle)
            self.create_model(hyperparams.architecture)
            self.create_optimizer(lr=hyperparams.lr)
            self.model = self.model.to(self.device)
            tb = self.setup_tensorboard_basics(hyperparams)
            print(f"Starting training, architecture={hyperparams.architecture}, lr={hyperparams.lr}, batch size={hyperparams.batch_size}, gamma={hyperparams.gamma}, shuffle={hyperparams.shuffle}, gamma_step={hyperparams.gamma_step} for {hyperparams.epoch_number} epochs")
            previous_mean_result = float_info.max
            previous_slope = float("-inf")

            for epoch in range(hyperparams.epoch_number):
                if not self.find_gamma_step:
                    if epoch % hyperparams.gamma_step == 0 and epoch > 0:
                        new_lr = hyperparams.lr * hyperparams.gamma
                        hyperparams = hyperparam_combination(new_lr, *hyperparams[1:])
                start = default_timer()
                result = self.train_valid_one_epoch(hyperparams.lr)
                stop = default_timer()
                mean_result = averaged_epoch_results(*map(mean, result))
                new_slope = ((previous_mean_result - mean_result.valid_loss_mean)/(epoch-1 - epoch))
                self.update_tensorboard_plots(tb, mean_result, epoch, hyperparams.lr)
                self.save_model_if_best(mean_result, hyperparams, epoch+1)
                graph_state = slope_tracker(epoch, previous_slope, new_slope)
                decayed_lr = self.decay_learning_rate(hyperparams, graph_state)
                print(f"Finished epoch {epoch+1} in {stop-start}s; train loss: {mean_result.train_loss_mean}, valid loss: {mean_result.valid_loss_mean}; train accuracy: {mean_result.train_accuracy_mean*100}%, valid_accuracy: {mean_result.valid_accuracy_mean*100}%")
                if decayed_lr:
                    print(f"Decayed lr - old learning rate = {hyperparams.lr}, new learning rate = {decayed_lr}, old slope = {previous_slope}, new slope = {new_slope}")
                    hyperparams = hyperparam_combination(decayed_lr, *hyperparams[1:])
                previous_mean_result = mean_result.valid_loss_mean
                previous_slope = new_slope                
            
            tb.close()
            print("Finished training\n" + "-" * 20 + "\n")
    
    def best_lr_for_hyperparameters(self):
        best_learning_rates = {}
        for batch_size, shuffle in product(*[self.hyperparameters["batch_size"], self.hyperparameters["shuffle"]]):
            best_lr = self.find_best_lr(batch_size, shuffle)
            best_learning_rates[(batch_size, shuffle)] = best_lr
        return best_learning_rates

    def find_best_lr(self, batch_size, shuffle=True) -> float:
        self.reproducible(seed=42)
        self.create_model()
        self.model.train()
        dl_train = fastai.data.load.DataLoader(self.train_dataset, bs=batch_size, shuffle=shuffle)
        dl_valid = fastai.data.load.DataLoader(self.valid_dataset, bs=batch_size, shuffle=shuffle)
        dls = fastai.data.core.DataLoaders(dl_train, dl_valid, device=self.device)
        learn = fastai.learner.Learner(dls, model=self.model, loss_func=self.loss_func)
        suggested_lr = learn.lr_find().lr_min
        return suggested_lr

    def make_dataloaders(self, batch_size:int=64, num_workers:int=1, shuffle:bool=True, mode="train"):
        if mode=="train":
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=shuffle)
        elif mode=="test":
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    def create_optimizer(self, lr:float, momentum:float=0.9):
        self.optimizer = self.optimizer_algorythm(self.model.parameters(), lr=lr, momentum=0.9)

    def setup_tensorboard_basics(self, hyperparams:namedtuple) -> SummaryWriter:
        tb = SummaryWriter(comment=f" {hyperparams.architecture} lr={hyperparams.lr} epochs={hyperparams.epoch_number} batch size={hyperparams.batch_size} gamma={hyperparams.gamma} gamma_step={hyperparams.gamma_step} shuffle={hyperparams.shuffle}")
        images, labels = next(iter(self.train_loader))
        grid = torchvision.utils.make_grid(images)
        tb.add_image("First_batch", grid)
        tb.add_graph(self.model, images.cuda())
        return tb    
    
    def train_valid_one_epoch(self, lr:float) -> Tuple[List[float], List[float]]:
        self.adjust_lr(lr)
        self.model.train()
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        train_batch_times = []
        valid_batch_times = []
        for batch_x, batch_y in self.train_loader:
            train_batch_start = default_timer()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred = self.model(batch_x)
            accuracy = self.accuracy(pred, batch_y)
            train_accuracies.append(accuracy)
            loss = self.loss_func(pred, batch_y)
            train_losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_batch_stop = default_timer()
            train_batch_times.append(train_batch_stop - train_batch_start)
            
        self.model.eval()    
        with torch.no_grad():
            for batch_x, batch_y in self.valid_loader:
                valid_batch_start = default_timer()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred = self.model(batch_x)
                valid_accuracy = self.accuracy(pred, batch_y)
                valid_accuracies.append(valid_accuracy)
                loss = self.loss_func(pred, batch_y)
                valid_losses.append(loss.item())
                valid_batch_stop = default_timer()
                valid_batch_times.append(valid_batch_stop - valid_batch_start)
        
        return self.results(train_losses, valid_losses, train_accuracies, valid_accuracies, train_batch_times, valid_batch_times)

    def adjust_lr(self, new_lr:float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
            lr = param_group["lr"]

    def accuracy(self, pred:torch.Tensor, batch_y:torch.Tensor):
        predicted_classes = torch.argmax(pred, dim=1)
        correct = (predicted_classes == batch_y).float().sum()
        accuracy = (correct/batch_y.shape[0]).item()
        return accuracy
    
    def update_tensorboard_plots(self, tb:SummaryWriter, mean_result:namedtuple, epoch:int, learning_rate:float) -> None:
        for plot_title, value in [("Train_loss", mean_result.train_loss_mean),
                                  ("Valid_loss", mean_result.valid_loss_mean),
                                  ("Train_accuracy", mean_result.train_accuracy_mean),
                                  ("Valid_accuracy", mean_result.valid_accuracy_mean),
                                  ("Train_batch_time", mean_result.train_batch_time_mean),
                                  ("Valid_batch_time", mean_result.valid_batch_time_mean),
                                  ("Train_learning_rate", learning_rate)]:
            tb.add_scalar(plot_title, value, epoch)
        
        for param_name, param in self.model.named_parameters():
            tb.add_histogram(param_name, param, epoch)
            tb.add_histogram(f"{param_name} gradient", param.grad, epoch)
    
    def save_model_if_best(self, mean_result:namedtuple, hyperparams:namedtuple, epoch:int) -> None:
        if mean_result.valid_loss_mean < self.best_run.valid_loss_mean:
            best_model = dumps(self.model)
            best_optimizer = dumps(self.optimizer)
            best_valid_loss_mean = mean_result.valid_loss_mean
            self.best_run = self.run(best_valid_loss_mean, best_model, best_optimizer, hyperparams, epoch)

    def decay_learning_rate(self, hyperparams, graph_state):
        if self.find_gamma_step:
            if graph_state.previous_slope/graph_state.new_slope < 0.95:
                new_lr = hyperparams.lr * hyperparams.gamma
                return new_lr
        elif graph_state.epoch+2 % hyperparams.gamma_step == 0:
            new_lr = hyperparams.lr * hyperparams.gamma
            return new_lr

    def test(self) -> None:
        self.model = loads(self.best_run.model)
        self.make_dataloaders(batch_size=self.best_run.hyperparams.batch_size,
                                shuffle=False,
                                mode="test")
        self.model.eval()
        print(f"Starting testing, model from epoch number {self.best_run.epoch}, architecture={self.best_run.hyperparams.architecture}, lr={self.best_run.hyperparams.lr}, batch_size={self.best_run.hyperparams.batch_size}, gamma={self.best_run.hyperparams.gamma}, gamma_step={self.best_run.hyperparams.gamma_step}")
        test_losses = []
        test_accuracies = []
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred = self.model(batch_x)
                test_accuracy = self.accuracy(pred, batch_y)
                test_accuracies.append(test_accuracy)
                loss = self.loss_func(pred, batch_y)
                test_losses.append(loss.item())
        test_loss_mean = mean(test_losses)
        test_accuracy_mean = mean(test_accuracies)
        print(f"Finished testing, testing loss: {test_loss_mean}, test accuracy: {test_accuracy_mean}\n" + "-" * 20)

    def write_best_model(self, path):
        print(f"Writing best model from epoch number {self.best_run.epoch}, lr={self.best_run.hyperparams.lr}, batch_size={self.best_run.hyperparams.batch_size}, gamma={self.best_run.hyperparams.gamma}, gamma_step={self.best_run.hyperparams.gamma_step}...")
        torch.save(self.best_run.model, path)
        print("Done.")
