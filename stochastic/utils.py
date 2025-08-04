import torch

from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import Callable
from typing import Callable


def dataloader_desc(dataloader : DataLoader) -> str:
    return "\n".join([f"{dataloader.batch_size=}", f"{dataloader.num_workers=}", f"{dataloader.dataset=}"])

def eval_loss(net : Module, dataloader : DataLoader, device : torch.device, criterion : Callable[[Tensor, Tensor], Tensor]) -> float:
    net.eval()
    loss = 0.
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).cpu().item() / len(dataloader)
    return loss

def eval_accuracy(net : Module, dataloader : DataLoader, device : torch.device) -> float:
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0:
        total += 1e-16
        print("Division by zero encountered")
    return correct / total

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device

def eavg(arr, alpha):
    ret = arr[0]
    for i in range(1, len(arr)):
        ret = alpha * arr[i] + (1 - alpha) * ret
    return ret