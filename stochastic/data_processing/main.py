import os
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from util import plot_setup


def search(line):
    # look for one of the algs below in the current line
    algs = ['SGD', 'Adam', 'SGDm', 'adaptiveNPGM']
    for a in algs:
        if a in line:
            return a
    return None


def parse_file(name):
    pattern = r"epoch=(\d+), loss_train=([0-9.eE+-]+), loss_test=([0-9.eE+-]+), acc_train=([0-9.eE+-]+), acc_test=([0-9.eE+-]+)"
    loss_train = []
    acc_test = []
    opt = None

    with open(name, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines:
            # look for alg in line
            alg = search(line)
            if alg is not None:
                if opt is not None:
                    raise Exception("Multiple algorithms found")
                opt = alg

            # try to match the pattern
            m = re.match(pattern, line)
            if m is not None:
                loss_train.append(float(m.group(2)))
                acc_test.append(float(m.group(5)))
    print(len(loss_train), len(acc_test))
    return opt, (loss_train, acc_test)


def calc_stats(lists):
    max_len = max(len(l) for l in lists)
    padded = [np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in lists]
    stacked = np.vstack(padded)
    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def calc_stats_log(lists):
    max_len = max(len(l) for l in lists)
    padded = [np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in lists]
    stacked = np.vstack(padded)
    return np.nanmean(np.log(stacked), axis=0), np.nanstd(np.log(stacked), axis=0)


def main():
    os.makedirs('saved_figs', exist_ok=True)
    data = defaultdict(list)
    target = 'resnet'
    target_folder = f'./{target}_data'
    for filename in os.listdir(target_folder):
        filepath = os.path.join(target_folder, filename)
        alg, dat = parse_file(filepath)
        data[alg].append(dat)

    mp = {'SGD': 0, 'Adam': 1, 'SGDm': 2, 'adaptiveNPGM': 3}
    labels = [r'$\textnormal{SGD}$', r'$\textnormal{Adam}$', r'$\textnormal{SGDm}$', r'$\textnormal{adaptiveNPGM}$']
    colors = ['red', 'black', 'darkorange', 'seagreen']
    markers = ['v', 's', 'x', 'D']
    data = {k: v for k, v in data.items() if k is not None}
    plot_setup()
    plt.figure(figsize=(8, 6))
    sorted_items = sorted(data.items(), key=lambda item: mp[item[0]])
    for i, (k, v) in enumerate(sorted_items):
        idx = mp[k]
        losses = [t[0] for t in v]
        mean, std = calc_stats_log(losses)
        xs = range(1, len(mean) + 1)
        plt.semilogy(xs, np.exp(mean), label=labels[idx], color=colors[idx], marker=markers[i], markevery=15)
        plt.fill_between(xs, np.exp(mean - std), np.exp(mean + std), color=colors[idx], alpha=0.3)
    plt.xlabel(r'$\textnormal{Epochs}$')
    plt.ylabel(r'$\textnormal{Training loss}$')
    plt.legend(loc='upper right')
    plt.savefig(f'saved_figs/{target}_train_loss.pdf', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(8, 6))
    for i, (k, v) in enumerate(sorted_items):
        idx = mp[k]
        losses = [t[1] for t in v]
        mean, std = calc_stats(losses)
        xs = range(1, len(mean) + 1)
        plt.plot(xs, 100 * mean, label=labels[idx], color=colors[idx], marker=markers[i], markevery=15)
        plt.fill_between(xs, 100 * (mean - std), 100 * (mean + std), color=colors[idx], alpha=0.3)
    plt.xlabel(r'$\textnormal{Epochs}$')
    plt.ylabel(r'$\textnormal{Test accuracy}$')
    plt.legend(loc='lower right')
    plt.ylim((60, None))
    plt.savefig(f'saved_figs/{target}_test_acc.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    # plt.close('all')


if __name__ == '__main__':
    main()