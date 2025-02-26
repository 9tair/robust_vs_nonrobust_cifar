import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

class CIFAR10Loader:
    def __init__(self, data_dir="./data", batch_size=128):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    
    def get_loaders(self):
        train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
        test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
