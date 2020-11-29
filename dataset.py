import torch

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data:(list, list)):
        self.data = data
        assert (len(self.data[0]) == len(self.data[1])), "Passed lists are not equal in length"

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        return (self.data[0][idx], self.data[1][idx])
