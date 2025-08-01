import torch
import argparse
from problems.mnist import mnist_mlp
from problems.cifar_resnet import cifar_resnet18
from problems.wikitext import wikitext_transf
from problems.fashion_mnist import fashion_mnist_cnn
from grid_search import GridSearch1D
from optimizers.optimizers import *
from utils import *

def get_problem(name):
    name = name.lower()
    if 'mnist' in name:
        print("[INFO] mnist selected")
        return mnist_mlp
    elif 'fashion' in name:
        print("[INFO] fashion mnist selected")
        return fashion_mnist_cnn
    elif 'resnet' in name:
        print("[INFO] resnet selected")
        return cifar_resnet18
    elif 'wikitext' in name:
        print("[INFO] wikitext selected")
        return wikitext_transf
    else:
        raise ValueError(f"Unknown problem type: {name}")

def get_optimizer(name, problem, momentum):
    name = name.lower()
    if 'iso' in name:
        print("iso selected")
        return GridSearch1D(lambda gamma: problem(
            get_device(), lambda params: PrecGDIso(params, Cosh(), gamma, 1, momentum)),
            [1e-3, 1e-2, 1e-1])
    elif 'sep' in name:
        print("sep selected")
        return GridSearch1D(lambda gamma: problem(
            get_device(), lambda params: PrecGDSep(params, Cosh(), gamma, 1, momentum)),
            [1e-3, 1e-2, 1e-1])
    elif 'adam' in name:
        print("adam selected")
        return GridSearch1D(lambda lr: problem(
            get_device(), lambda params: torch.optim.Adam(params, lr=lr)),
            [1e-3, 1e-2, 1e-1])
    elif 'sgd' in name:
        print("sgd selected")
        return GridSearch1D(lambda lr: problem(
            get_device(), lambda params: torch.optim.SGD(params, lr=lr, momentum=momentum)),
            [1e-3, 1e-2, 1e-1])
    else:
        raise ValueError(f"Unknown optimizer type: {name}")

def main():
    parser = argparse.ArgumentParser(description="Run experiment with selected problem and optimizer.")
    parser.add_argument('--problem', type=str, default='resnet', help="Problem to solve (e.g., mnist, resnet, wikitext, fashion)")
    parser.add_argument('--optimizer', type=str, default='iso', help="Optimizer to use (e.g., iso, sep, adam, sgd)")
    parser.add_argument('--momentum', type=float, default='0.9', help="Momentum")

    args = parser.parse_args()
    print(args)

    problem = get_problem(args.problem)
    search = get_optimizer(args.optimizer, problem, args.momentum)

    best_score, best_val = search.search()
    print(best_score, best_val)

if __name__ == '__main__':
    main()