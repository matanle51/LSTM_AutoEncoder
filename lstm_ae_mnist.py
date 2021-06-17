import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from models.LSTMAE import LSTMAE
from models.LSTMAE_CLF import LSTMAECLF
from train_utils import train_model, eval_model


parser = argparse.ArgumentParser(description='LSTM_AE MNIST TASK')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N', help='LSTM hidden state size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
parser.add_argument('--input-size', type=int, default=28, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0.0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', default=None, metavar='GC', help='gradient clipping value; if None no clipping')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--model-type', default='LSTMAE_CLF', help='type of model to use: LSTMAE or LSTMAE_CLF')
parser.add_argument('--seq-len', default=28, help='sequence full length')
parser.add_argument('--kernel-size', default=(1, 1), help='Kernel size for LSTM Conv AE')
parser.add_argument('--n-classes', default=10, help='number of classes in case of a lstm ae with clf')
parser.add_argument('--scheduler', default=None, help='Learning rate scheduler type. if None - no lr scheduling')


args = parser.parse_args(args=[])

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model_type in ('LSTMAE', 'LSTMAE_CLF'):
    transforms_lst = transforms.Compose([transforms.ToTensor(),
                                         # transforms.Normalize((0.1307,), (0.3081,)),
                                         transforms.Lambda(lambda x: x.squeeze())])
else:
    transforms_lst = transforms.Compose([transforms.ToTensor(),
                                         # transforms.Normalize((0.1307,), (0.3081,)),
                                         transforms.Lambda(lambda x: x.reshape(-1, 1, 1, 1))])

# setup data loaders
train_set = datasets.MNIST('data', train=True, download=True, transform=transforms_lst)
train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

train_iter = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
val_iter = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

test_iter = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms_lst),
                                        batch_size=args.batch_size, shuffle=False, **kwargs)


def main():
    """
    Function to train and evaluate LSTM-AE and LSTM-AE classifier on MNIST dataset.
    """
    # Load model
    model = create_model()

    # Create optimizer, loss function and scheduler policy
    optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = mse_ce_loss if args.model_type == 'LSTMAE_CLF' else torch.nn.MSELoss(reduction='sum')
    scheduler = get_scheduler(optimizer)

    train_acc_lst, train_loss_lst, val_acc_lst, val_loss_lst = [], [], [], []
    for epoch in range(args.epochs):
        # Train loop
        train_loss, train_acc, _ = train_model(criterion, epoch, model, args.model_type, optimizer, train_iter, args.batch_size, args.grad_clipping, args.log_interval, scheduler)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        if (epoch + 1) % (int(args.epochs / 10)) == 0:
            plot_images(model, val_iter, description=args.model_type, num_to_plot=3)
        # Evaluate
        val_loss, val_acc = eval_model(criterion, model, args.model_type, val_iter, mode='Validation')
        val_acc_lst.append(val_acc)
        val_loss_lst.append(val_loss)

    # Run on test set
    plot_images(model, test_iter, description=args.model_type, num_to_plot=3)
    eval_model(criterion, model, args.model_type, test_iter, mode='Test')

    # plot accuracy and loss graphs
    plot_loss_acc(train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst, description=args.model_type)

    # Save model
    save_model(model)


def mse_ce_loss(data_rec, data, out_labels, labels):
    """
    Function that calculated both reconstruction and classification losses
    :param data_rec: reconstructed MNIST figure
    :param data: actual MNIST figure
    :param out_labels: predicted labels
    :param labels: actual labels
    :return: the MSE loss on the reconstruction and CE loss on the classification
    """
    mse_loss = torch.nn.MSELoss(reduction='sum')(data_rec, data)
    ce_loss = torch.torch.nn.CrossEntropyLoss(reduction='sum')(out_labels, labels)

    return mse_loss, ce_loss


def plot_loss_acc(train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst, description):
    """
    Plots both accuracy (train and validation) and loss (train and validation) graphs.
    :param train_loss_lst: list of all train losses recorded from the training process
    :param train_acc_lst: list of all train accuracy recorded from the training process
    :param val_loss_lst: list of all validation losses recorded from the training process
    :param val_acc_lst: list of all validation accuracy recorded from the training process
    :param description: additional description to add to the plots
    :return:
    """
    epochs = [t for t in range(len(train_loss_lst))]

    plt.figure()
    plt.plot(epochs, train_acc_lst, color='r', label='Train accuracy')
    plt.plot(epochs, val_acc_lst, color='g', label='Test accuracy')
    plt.xticks([i for i in range(0, len(epochs)+1, int(len(epochs)/10))])
    plt.xlabel('Epochs')
    plt.ylabel(f'{description} Accuracy value')
    plt.legend()
    plt.title(f'{description} accuracy vs. epochs')
    plt.savefig(f'{description}_acc_graphs.png')
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss_lst, color='r', label='Train loss')
    plt.plot(epochs, val_loss_lst, color='g', label='Test loss')
    plt.xticks([i for i in range(0, len(epochs)+1, int(len(epochs)/10))])
    plt.xlabel('Epochs')
    plt.ylabel(f'{description} Loss value')
    plt.legend()
    plt.title(f'{description} loss vs. epochs')
    plt.savefig(f'{description}_loss_graphs.png')
    plt.show()


def save_model(model):
    """
    Save model's state_dict for future use
    :param model: model to save
    :return:
    """
    torch.save(model.state_dict(),
               os.path.join(args.model_dir, f'model={args.model_type}_hs={args.hidden_size}_bs={args.batch_size}'
                                            f'_epochs={args.epochs}_clip={args.grad_clipping}.pt'))


def get_scheduler(optimizer):
    """
    Retrieve the desired learning rate scheduler
    :param optimizer: optimizer for which we attach the scheduler
    :return: pytorch scheduler
    """
    if args.scheduler == 'OneCycleLR':
        print(f'Using {args.scheduler} LR scheduler')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_iter),
                                                        epochs=args.epochs, verbose=False)
    else:
        print(f'No LR scheduler is used')
        scheduler = None
    return scheduler


def create_model():
    """
    Create the desired model - regular LSTM AE for reconstruction or LSTM AE for both reconstruction and classification.
    :return: model
    """
    if args.model_type == 'LSTMAE':
        print('Creating LSTM AE model')
        model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                       seq_len=args.seq_len)
    elif args.model_type == 'LSTMAE_CLF':
        print('Creating LSTM AE with classifier')
        model = LSTMAECLF(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                          n_classes=args.n_classes, seq_len=args.seq_len)
    else:
        raise NotImplementedError(f'Selected model type is not implemented: {args.model_type}')
    model = model.to(device)
    return model


def plot_images(model, test_iter, description, num_to_plot=3):
    """
    Plot the reconstructed vs. Original MNIST figures
    :param model: model trained to reconstruct MNIST figures
    :param test_iter: test data loader
    :param description: additional description for the plots
    :param num_to_plot: number of random plots to present
    :return:
    """
    model.eval()
    data = next(iter(test_iter))[0].to(device)
    if args.model_type == 'LSTMAE_CLF':
        rec_data, _ = model(data)
    else:
        rec_data = model(data)

    for i in range(num_to_plot):
        orig = data[i].reshape(28, 28).to('cpu').detach().numpy()  # MNIST are 28*28 grey scale images
        plt.imshow(orig, cmap="gray")
        plt.title(f'{args.model_type} Original image #{i + 1}')
        plt.savefig(f'{args.model_type} Original image #{i + 1}.png')
        plt.show()

        plt.imshow(rec_data[i].reshape(28, 28).to('cpu').detach().numpy(), cmap="gray")
        plt.title(f'{args.model_type} Reconstructed image #{i + 1} for {description}')
        plt.savefig(f'{args.model_type} Reconstructed image #{i + 1} for {description}.png')
        plt.show()


if __name__ == '__main__':
    main()
