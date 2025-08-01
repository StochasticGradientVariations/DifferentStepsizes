import torch

from dataloaders.mnist_loader import load_mnist
from problems.problem import Problem
from models.simple_nn import SimpleNN
# from models.simple_cnn import SimpleMNISTCNN
from utils import eval_accuracy

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from typing import Callable, Iterator

def mnist_mlp(device : torch.device, optimizer : Callable[[Iterator[Parameter]], Optimizer], seed=0, batch_size=256):
    torch.manual_seed(seed)
    prob = Problem("mnist", "MNIST (digits) simple NN", device, seed)
    trainloader, testloader, _ = load_mnist(batch_size=batch_size, num_workers=4, data_seed=seed)
    model = SimpleNN(input_size=28*28, output_size=10, hidden_layers=[512, 256])
    model.to(device)

    prob.set_data(trainloader, testloader)
    prob.set_model(model)
    prob.set_optimizer(optimizer)
    prob.set_loss_fn(torch.nn.CrossEntropyLoss())
    prob.set_acc_fn(eval_accuracy)
    return prob


def mnist_deep_mlp(device : torch.device, optimizer : Callable[[Iterator[Parameter]], Optimizer], seed=0, batch_size=256):
    torch.manual_seed(seed)
    prob = Problem("mnist", "MNIST (digits) simple NN deep", device, seed)
    trainloader, testloader, _ = load_mnist(batch_size=batch_size, num_workers=4, data_seed=seed)
    model = SimpleNN(input_size=28*28, output_size=10, hidden_layers=[512, 256, 128, 128, 64])
    model.to(device)

    prob.set_data(trainloader, testloader)
    prob.set_model(model)
    prob.set_optimizer(optimizer)
    prob.set_loss_fn(torch.nn.CrossEntropyLoss())
    prob.set_acc_fn(eval_accuracy)
    return prob