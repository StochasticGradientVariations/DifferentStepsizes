import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from os import path
from config import DATA_DIR

def load_fashion_mnist(batch_size=128, num_workers=4, data_seed=0):
    torch.manual_seed(data_seed)
    num_classes = 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3520,))
    ])
    data_folder = path.join(DATA_DIR, 'data')
    trainloader = DataLoader(datasets.FashionMNIST(root=data_folder, train=True, download=True,
                                                transform=transform), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(datasets.FashionMNIST(root=data_folder, train=False, download=True,
                                   transform=transform), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader, testloader, num_classes