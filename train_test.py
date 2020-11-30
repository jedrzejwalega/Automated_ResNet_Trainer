import torch
import torch.nn as nn
import model
import torch.optim as optim
import input_data
import dataset
from typing import List,Tuple

class RunManager():
    def __init__(self, learning_rates:List[float], epochs:List[int], batch_size:int=64, gamma:float=0.1):
        self.model = None
        self.batch_size = batch_size
        self.learning_rates = learning_rates
        self.optimizer = None
        self.epochs = epochs
        self.train_data = None
        self.test_data = None
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.loss_func = nn.CrossEntropyLoss()
    
    def make_model(self, out_activations:int, in_channels:int=1, optimizer=optim.SGD):
        self.model = model.ResNet50(out_activations, in_channels)
        self.optimizer = optimizer(self.model.parameters(), lr=0.01, momentum=0.9)

    def pass_datasets(self, train_set:Tuple[torch.FloatTensor, torch.LongTensor], test_set:[Tuple[torch.FloatTensor, torch.LongTensor]]):
        train_set = dataset.ImageDataset(train_set)
        test_set = dataset.ImageDataset(test_set)
        train_len = len(train_set)
        lengths = [int(train_len*0.8), int(train_len*0.2)]
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(train_set, lengths=lengths)
        self.test_dataset = test_set
    
    def make_dataloaders(self, num_workers:int=1, shuffle:bool=True):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(self):
        self.model = self.model.to(self.device)
        for lr in self.learning_rates:
            for epoch_number in self.epochs:
                print(f"Starting training, lr={lr}")
                for epoch in range(epoch_number):
                    train_losses, valid_losses = self.__train_valid_one_epoch(lr)
                    print(f"Finished {epoch+1} epoch")
                print(f"Finished training of lr={lr} in {epoch_number} epochs")

    def __train_valid_one_epoch(self, lr:float) -> (list,list):
        self.__adjust_lr(lr)
        torch.manual_seed(42)
        self.model.train()
        train_losses = []
        valid_losses = []
        batch_num = 1
        for batch_x, batch_y in self.train_loader:

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred = torch.softmax(self.model(batch_x), dim=-1)
            loss = self.loss_func(pred, batch_y)
            train_losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()    
        for batch_x, batch_y in self.valid_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred = torch.softmax(self.model(batch_x), dim=-1)
            loss = self.loss_func(pred, batch_y)
            valid_losses.append(loss.item())
        
        return train_losses, valid_losses
    
    def __adjust_lr(self, new_lr:float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

input_data.download_mnist("/home/jedrzej/Desktop/fmnist")
train_images, train_labels = input_data.load_mnist("/home/jedrzej/Desktop/fmnist")
test_images, test_labels = input_data.load_mnist("/home/jedrzej/Desktop/fmnist")
program = RunManager([1], [3])
program.make_model(10)
program.pass_datasets((train_images, train_labels), (test_images, test_labels))
program.make_dataloaders()
program.train()
