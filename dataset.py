import torch

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data:(list, list), transform=None):
        self.data = data
        self.transform = transform
        assert (len(self.data[0]) == len(self.data[1])), "Passed lists are not equal in length"

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        item = self.data[0][idx]
        label = self.data[1][idx]
        if self.transform:
            item = self.transform(item)
        return (item, label)
