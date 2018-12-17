import torch
import torchvision
import pandas as pd

class DatasetCelebA(torch.utils.data.Dataset):
    def __init__(self, root, attr):
        self.dataset = torchvision.datasets.ImageFolder(root=root,
            transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()]))
        self.attr = pd.read_csv(attr)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.dataset[i][0], torch.from_numpy((
            self.attr.iloc[i].values[1:].astype(float) + 1) / 2).float()