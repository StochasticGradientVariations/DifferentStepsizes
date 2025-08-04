from  __future__ import annotations

import torch
import datetime
import numpy as np

from os import path, environ
from utils import dataloader_desc
from uuid import uuid4
from pathlib import Path
from utils import eval_loss
from copy import deepcopy
from config import DATA_DIR

# Imports for typing
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from typing import Callable, Iterator

class Problem:
    #######################################################
    #                      SETUP                          #
    #######################################################
    def __init__(self, output_folder : str, desc : str, device : torch.device, seed : int):
        self.output_folder : str = output_folder
        self.desc : str = desc
        self.device : torch.device = device
        self.seed : int = seed

        self.dataloader_train : DataLoader | None = None
        self.dataloader_test : DataLoader | None = None
        self.model : Module | None = None
        self.optimizer : Optimizer | None = None
        self.loss_fn : Callable[[Tensor, Tensor], Tensor] | None = None
        self.acc_fn : Callable[[Module, DataLoader, torch.device], float] | None = None
        self.losses : list[float] = []

        # Make result folder if not already existing
        folder_path = path.join(DATA_DIR, "results", f'{self.output_folder}')
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        self.file_path = path.join(folder_path, f'{str(datetime.datetime.now()).replace(" ", "_") + "___" + str(uuid4())}')
        self.epochs_trained = 0

    def set_data(self, dataloader_train : DataLoader, dataloader_test : DataLoader):
        assert self.dataloader_train is None, "Dataloader train has already been set"
        assert self.dataloader_test is None, "Dataloader test has already been set"
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

    def set_model(self, model : Module):
        assert self.model is None, "Model has already been set"
        self.model = model

    def set_optimizer(self, optimizer : Callable[[Iterator[Parameter]], Optimizer]):
        assert self.model is not None, "Model required"
        assert self.optimizer is None, "Optimizer has already been set"
        self.optimizer = optimizer(self.model.parameters())

    def set_loss_fn(self, loss_fn : Callable[[Tensor, Tensor], Tensor]):
        assert self.loss_fn is None, "Loss function has already been set"
        self.loss_fn = loss_fn

    def set_acc_fn(self, acc_fn : Callable[[Module, DataLoader, torch.device], float]):
        assert self.acc_fn is None, "Accuracy function has already been set"
        self.acc_fn = acc_fn

    #######################################################
    #                   TRAINING                          #
    #######################################################
    def train(self, epochs : int = 200):
        assert self.dataloader_train is not None, "Dataloader train not set"
        assert self.dataloader_test is not None, "Dataloader test not set"
        assert self.model is not None, "Model not set"
        assert self.optimizer is not None, "Optimizer not set"
        assert self.loss_fn is not None, "Loss function not set"

        print(f"[INFO] STARTING TRAINING\n{self.desc}")
        print(self.seed)
        with open(self.file_path, 'a') as f:
            if self.epochs_trained == 0:
                print(str(self) + "\n" + "=" * 150, file=f, flush=True)
            # Train loop
            while self.epochs_trained < epochs:
                self.epochs_trained += 1
                # One train epoch
                self.model.train()
                for inputs, outputs in self.dataloader_train:
                    inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                    self.optimizer.zero_grad()
                    pred = self.model(inputs)
                    loss = self.loss_fn(pred, outputs)
                    loss.backward()
                    self.optimizer.step()

                # Compute losses and accuracies
                loss_train = eval_loss(self.model, self.dataloader_train, self.device, self.loss_fn)
                acc_train = self.acc_fn(self.model, self.dataloader_train, self.device) if self.acc_fn else -1
                loss_test = eval_loss(self.model, self.dataloader_test, self.device, self.loss_fn) # do not use for parameter tuning
                acc_test = self.acc_fn(self.model, self.dataloader_test, self.device) if self.acc_fn else -1
                self.losses.append(Tensor.cpu(loss).detach().numpy())
                if np.isnan(self.losses[-1]):
                    print("[INFO] nan detected, stopping early")
                    return deepcopy(self.losses)
                print(f"After epoch={self.epochs_trained}: loss train={loss_train:.10f}, loss_test={loss_test:.10f}, acc_train={acc_train:.10f}, acc_test={acc_test:.10f}")
                print(f"epoch={self.epochs_trained}, {loss_train=}, {loss_test=}, {acc_train=}, {acc_test=}", file=f, flush=True)
        return deepcopy(self.losses)

    def __repr__(self):
        return "\n".join([self.desc, f"{self.device=}", f"{self.optimizer=}", 
                            f"{self.model=}", f"{self.loss_fn=}", f"{dataloader_desc(self.dataloader_train)}", f"{self.seed=}"])