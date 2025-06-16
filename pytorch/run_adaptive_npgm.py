import numpy as np
import torch
import time
import argparse

from optimizer_adaptive_npgm import AdaptiveNPGM
from utils import load_data, accuracy_and_loss, save_results, seed_everything
from resnet import ResNet18

def run_adaptive_npgm(
    net,
    trainloader,
    testloader,
    device,
    N_train,
    n_epoch=2,
    weight_decay=0.0,
    checkpoint=None,
    batch_size=128,
    noisy_train_stat=True
):
    losses, train_acc = [], []
    test_losses, test_acc = [], []
    it_train, it_test = [], []
    lrs, grad_norms = [], []

    net.to(device)
    net.train()

    criterion = torch.nn.CrossEntropyLoss()
    initial_lr = 1e-1
    # optimizer = AdaptiveNPGM(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    optimizer = AdaptiveNPGM(
        net.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
        r_k=2  # ← εδώ βαζουμε το damping factor που θελουμε
    )

    if checkpoint is None:
        checkpoint = len(trainloader) // 3

    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            # reset gradients
            optimizer.zero_grad(set_to_none=True)

            # closure: μόνο forward+backward
            def closure():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            # ένα call: forward, backward & adaptive update
            loss = optimizer.step(closure)

            # Logging (μία φορά)
            if i % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch:2d} | Step {i:4d}] loss = {loss.item():.4f}, lr = {lr:.2e}")

            elapsed = time.time() - start_time
            print(f"Minibatch {i:4d} took {elapsed:.4f}s")

            running_loss += loss.item()
            if i % 10 == 0:
                if noisy_train_stat:
                    losses.append(loss.item())
                    it_train.append(epoch + i * batch_size / N_train)
                lrs.append(optimizer.param_groups[0]['lr'])
                grad_norms.append(sum(
                    p.grad.data.norm().item() for p in net.parameters() if p.grad is not None
                ))

            if (i + 1) % checkpoint == 0:
                avg = running_loss / checkpoint
                tag = f"{avg:.4f}" if avg < 0.01 else f"{avg:.3f}"
                print(f"[{epoch+1}, {(i+1):5d}] loss: {tag}", end='')
                running_loss = 0.0

                ta, tl = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(ta)
                test_losses.append(tl)
                it_test.append(epoch + i * batch_size / N_train)
                net.train()

        if not noisy_train_stat:
            it_train.append(epoch)
            ta, tl = accuracy_and_loss(net, trainloader, device, criterion)
            train_acc.append(ta)
            losses.append(tl)
            net.train()

    return (
        np.array(losses),
        np.array(test_losses),
        np.array(train_acc),
        np.array(test_acc),
        np.array(it_train),
        np.array(it_test),
        np.array(lrs),
        np.array(grad_norms),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_decay',   type=float, default=0.)
    parser.add_argument('--batch_size',     type=int,   default=128)
    parser.add_argument('--n_epoch',        type=int,   default=120)
    parser.add_argument('--n_seeds',        type=int,   default=1)
    parser.add_argument('--output_folder',  type=str,   default='./')
    parser.add_argument('--noisy_train_stat', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_train = 50000
    trainloader, testloader, _ = load_data(batch_size=args.batch_size)
    checkpoint = len(trainloader) // 3

    for seed in range(args.n_seeds):
        seed_everything(42 + seed)
        net = ResNet18().to(device)
        (
            losses, test_losses, train_acc, test_acc,
            it_train, it_test, lrs, grad_norms
        ) = run_adaptive_npgm(
            net,
            trainloader,
            testloader,
            device,
            N_train,
            n_epoch=args.n_epoch,
            weight_decay=args.weight_decay,
            checkpoint=checkpoint,
            batch_size=args.batch_size,
            noisy_train_stat=args.noisy_train_stat
        )
        save_results(
            losses, test_losses, train_acc, test_acc,
            it_train, it_test, lrs=lrs, grad_norms=grad_norms,
            method="adaptive_npgm", experiment="cifar10_resnet18",
            folder=args.output_folder
        )
