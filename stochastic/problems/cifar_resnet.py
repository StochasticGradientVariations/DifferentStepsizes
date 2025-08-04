import torch

from dataloaders.cifar_loader import load_cifar
from problems.problem import Problem
from models.resnet import ResNet18
from utils import eval_accuracy

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from typing import Callable, Iterator

def cifar_resnet18(device : torch.device, optimizer : Callable[[Iterator[Parameter]], Optimizer], seed=0, batch_size=128):
    torch.manual_seed(seed)
    prob = Problem("cifar10", "CIFAR10 resnet 18", device, seed)
    trainloader, testloader, _ = load_cifar(batch_size=batch_size, num_workers=4, data_seed=0)
    model = ResNet18()
    model.to(device)

    prob.set_data(trainloader, testloader)
    prob.set_model(model)
    prob.set_optimizer(optimizer)
    prob.set_loss_fn(torch.nn.CrossEntropyLoss())
    prob.set_acc_fn(eval_accuracy)
    return prob