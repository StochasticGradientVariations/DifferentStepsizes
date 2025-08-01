from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from os import path
from config import DATA_DIR
import torch

def load_cifar(dataset='cifar10', batch_size=128, num_workers=4, data_seed=0):
    torch.manual_seed(data_seed)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_folder = path.join(DATA_DIR, 'data')
    if dataset == 'cifar10':
        num_classes = 10
        trainset = datasets.CIFAR10(root=data_folder, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=data_folder, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        num_classes = 100
        trainset = datasets.CIFAR100(root=data_folder, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=data_folder, train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Only cifar 10 and cifar 100 are currently supported')

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, num_classes