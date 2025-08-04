import torch

from dataloaders.fashion_mnist_loader import load_fashion_mnist
from problems.problem import Problem
from models.simple_cnn import SimpleMNISTCNN
from utils import eval_accuracy

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from typing import Callable, Iterator

def fashion_mnist_cnn(device : torch.device, optimizer : Callable[[Iterator[Parameter]], Optimizer], seed=0, batch_size=256):
    torch.manual_seed(seed)
    prob = Problem("fashion-mnist", "fashion MNIST cnn", device, seed)
    trainloader, testloader, classes = load_fashion_mnist(batch_size=batch_size, num_workers=4, data_seed=seed)
    model = SimpleMNISTCNN(in_channels=1, num_classes=classes)
    model.to(device)

    prob.set_data(trainloader, testloader)
    prob.set_model(model)
    prob.set_optimizer(optimizer)
    prob.set_loss_fn(torch.nn.CrossEntropyLoss())
    prob.set_acc_fn(eval_accuracy)
    return prob