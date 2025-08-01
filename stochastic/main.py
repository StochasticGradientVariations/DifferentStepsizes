import torch
import argparse
# from problems.mnist import mnist_mlp
from problems.cifar_resnet import cifar_resnet18
# from problems.wikitext import wikitext_transf
# from problems.ptb_transformer import ptb_transf
# from problems.fashion_mnist import fashion_mnist_cnn
# from problems.cifar100 import cifar100
# from grid_search import GridSearch1D
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
    elif 'ptb' in name:
        print("[INFO] ptb selected")
        return ptb_transf
    elif 'cifar100' in name:
        print("[INFO] cifar100 selected")
        return cifar100
    else:
        raise ValueError(f"Unknown problem type: {name}")

def train(name, problem, momentum, lr, epochs):
    name = name.lower()
    return problem(get_device(), lambda params: AdaptiveNPGM(params, lr=lr), seed = 2023).train(epochs=epochs)
    
    
def main():
    parser = argparse.ArgumentParser(description="Run experiment with selected problem and optimizer.")
    parser.add_argument('--problem', type=str, default='resnet', help="Problem to solve (e.g., mnist, resnet, wikitext, ptb, fashion, cifar100)")
    parser.add_argument('--optimizer', type=str, default='iso', help="Optimizer to use (e.g., iso, sep, adam, sgd)")
    parser.add_argument('--momentum', type=float, default='0', help="Momentum param")
    parser.add_argument('--lr', type=float, default='1', help="Stepsize gamma")
    parser.add_argument('--epochs', type=int, default='200', help="Epochs")
    # parser.add_argument('--epochs', type=int, default='200', help="Epochs")

    args = parser.parse_args()
    print(args)
    problem = get_problem(args.problem)
    train(args.optimizer, problem, args.momentum, args.lr, args.epochs)

if __name__ == '__main__':
    main()