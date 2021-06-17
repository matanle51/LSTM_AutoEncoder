import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.LSTMAE import LSTMAE
from train_utils import train_model, eval_model

parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=256, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--input-size', type=int, default=1, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0.0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=None, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-type', default='LSTMAE', help='currently only LSTMAE')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--seq-len', default=50, help='sequence full size')
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')

args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# folder settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)


class toy_dataset(torch.utils.data.Dataset):
    def __init__(self, toy_data):
        self.toy_data = toy_data

    def __len__(self):
        return self.toy_data.shape[0]

    def __getitem__(self, index):
        return self.toy_data[index]


def main():
    # Create data loaders
    train_iter, val_iter, test_iter = create_dataloaders(args.batch_size)

    # Create model
    model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout, seq_len=args.seq_len)
    model.to(device)

    # Create optimizer & loss functions
    optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.MSELoss(reduction='sum')

    # Grid search run if run-grid-search flag is active
    if args.run_grid_search:
        hyper_params_grid_search(train_iter, val_iter, criterion)
        return

    # Train & Val
    for epoch in range(args.epochs):
        # Train loop
        train_model(criterion, epoch, model, args.model_type, optimizer, train_iter, args.batch_size, args.grad_clipping,
                    args.log_interval)
        eval_model(criterion, model, args.model_type, val_iter)
    eval_model(criterion, model, args.model_type, test_iter, mode='Test')

    # Save model
    torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_hs={args.hidden_size}_bs={args.batch_size}'
                                                                f'_epochs={args.epochs}_clip={args.grad_clipping}.pt'))

    # Plot original images and their corresponding reconstructed images
    plot_orig_vs_reconstructed(model, test_iter)


def create_toy_data(num_of_sequences=10000, sequence_len=50) -> torch.tensor:
    """
    Generate num_of_sequences random sequences with length of sequence_len each.
    :param num_of_sequences: number of sequences to generate
    :param sequence_len: length of each sequence
    :return: pytorch tensor containing the sequences
    """
    # Random uniform distribution
    toy_data = torch.rand((num_of_sequences, sequence_len, 1))

    return toy_data


def create_dataloaders(batch_size, train_ratio=0.6, val_ratio=0.2):
    """
    Build train, validation and tests dataloader using the toy data
    :return: Train, validation and test data loaders
    """
    toy_data = create_toy_data()
    len = toy_data.shape[0]

    train_data = toy_data[:int(len * train_ratio), :]
    val_data = toy_data[int(train_ratio * len):int(len * (train_ratio + val_ratio)), :]
    test_data = toy_data[int((train_ratio + val_ratio) * len):, :]

    print(f'Datasets shapes: Train={train_data.shape}; Validation={val_data.shape}; Test={test_data.shape}')
    train_iter = torch.utils.data.DataLoader(toy_dataset(train_data), batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(toy_dataset(val_data), batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(toy_dataset(test_data), batch_size=batch_size, shuffle=False)

    return train_iter, val_iter, test_iter


def plot_toy_data(toy_example, description, color='b'):
    """
    Recieves a toy raw data sequence and plot it
    :param toy_example: toy data example sequence
    :param description: additional description to the plot
    :param color: graph color
    :return:
    """
    time_lst = [t for t in range(toy_example.shape[0])]

    plt.figure()
    plt.plot(time_lst, toy_example.tolist(), color=color)
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    # plt.legend()
    plt.title(f'Single value vs. time for toy example {description}')
    plt.show()


def plot_orig_vs_reconstructed(model, test_iter, num_to_plot=2):
    """
    Plot the reconstructed vs. Original MNIST figures
    :param model: model trained to reconstruct MNIST figures
    :param test_iter: test data loader
    :param num_to_plot: number of random plots to present
    :return:
    """
    model.eval()
    # Plot original and reconstructed toy data
    plot_test_iter = iter(torch.utils.data.DataLoader(test_iter.dataset, batch_size=1, shuffle=False))

    for i in range(num_to_plot):
        orig = next(plot_test_iter).to(device)
        with torch.no_grad():
            rec = model(orig)

        time_lst = [t for t in range(orig.shape[1])]

        # Plot original
        plot_toy_data(orig.squeeze(), f'Original sequence #{i + 1}', color='g')

        # Plot reconstruction
        plot_toy_data(rec.squeeze(), f'Reconstructed sequence #{i + 1}', color='r')

        # Plot combined
        plt.figure()
        plt.plot(time_lst, orig.squeeze().tolist(), color='g', label='Original signal')
        plt.plot(time_lst, rec.squeeze().tolist(), color='r', label='Reconstructed signal')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.legend()
        title = f'Original and Reconstruction of Single values vs. time for toy example #{i + 1}'
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.show()


def hyper_params_grid_search(train_iter, val_iter, criterion):
    """
    Function to perform hyper-parameter grid search on a pre-defined range of values.
    :param train_iter: train dataloader
    :param val_iter: validation data loader
    :param criterion: loss criterion to use (MSE for reconstruction)
    :return:
    """
    lr_lst = [1e-2, 1e-3, 1e-4]
    hs_lst = [16, 32, 64, 128, 256]
    clip_lst = [None, 10, 1]

    total_comb = len(lr_lst) * len(hs_lst) * len(clip_lst)
    print(f'Total number of combinations: {total_comb}')

    curr_iter = 1
    best_param = {'lr': None, 'hs': None, 'clip_val': None}
    best_val_loss = np.Inf
    params_loss_dict = {}

    for lr in lr_lst:
        for hs in hs_lst:
            for clip_val in clip_lst:
                print(f'Starting Iteration #{curr_iter}/{total_comb}')
                curr_iter += 1
                model = LSTMAE(input_size=args.input_size, hidden_size=hs, dropout_ratio=args.dropout,
                               seq_len=args.seq_len)
                model = model.to(device)
                optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=lr, weight_decay=args.wd)

                for epoch in range(args.epochs):
                    # Train loop
                    train_model(criterion, epoch, model, args.model_type, optimizer, train_iter, args.batch_size, clip_val,
                                args.log_interval)
                avg_val_loss, val_acc = eval_model(criterion, model, args.model_type, val_iter)
                params_loss_dict.update({f'lr={lr}_hs={hs}_clip={clip_val}': avg_val_loss})
                if avg_val_loss < best_val_loss:
                    print(f'Found better validation loss: old={best_val_loss}, new={avg_val_loss}; parameters: lr={lr},hs={hs},clip_val={clip_val}')
                    best_val_loss = avg_val_loss
                    best_param = {'lr': lr, 'hs': hs, 'clip_val': clip_val}

    print(f'Best parameters found: {best_param}')
    print(f'Best Validation Loss: {best_val_loss}')
    print(f'Parameters loss: {params_loss_dict}')


if __name__ == '__main__':
    main()
