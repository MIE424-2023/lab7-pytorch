import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({round(100 * batch_idx / len(train_loader))}%)]\tLoss: {round(loss.item(), 4)}')
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f'\nTest set: Average loss: {round(test_loss, 4)}, Accuracy: {correct}/{len(test_loader.dataset)} ({round(100. * correct / len(test_loader.dataset))}%)\n')


def plot_helper(model, device, train_loader, test_loader):
    """
    Plot helper that recomputes loss and acccuracyc and returns it

    Parameters
    ----------
    model
    device
    train_loader
    test_loader

    Returns
    -------

    """
    model.eval()
    test_loss = 0
    train_loss = 0
    train_correct = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_correct += pred.eq(target.view_as(pred)).sum().item()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    train_loss /= len(train_loader.dataset)
    test_acc = 100. * test_correct / len(test_loader.dataset)
    train_acc = 100. * train_correct / len(train_loader.dataset)

    return train_loss, test_loss, train_acc, test_acc


def main(args: argparse.Namespace):
    # Training settings

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = getattr(datasets, args.dataset)('../data', train=True, download=True,
                                               transform=transform)
    dataset2 = getattr(datasets, args.dataset)('../data', train=False,
                                               transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    epochs = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        if args.plot:
            train_loss, test_loss, train_acc, test_acc = plot_helper(model, device, train_loader, test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epochs.append(epoch)

        scheduler.step()

    if args.plot:
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.legend()
        plt.savefig(f'Loss_lr={args.lr}_batch={args.batch_size}_optimizer={args.optimizer}.png')
        plt.clf()
        plt.plot(epochs, train_accs, label="Training Accuracy")
        plt.plot(epochs, test_accs, label="Test Accuracy")
        plt.legend()
        plt.savefig(f'Accuracy_lr={args.lr}_batch={args.batch_size}_optimizer={args.optimizer}.png')
    if args.save_model:
        torch.save(model.state_dict(), args.dataset + ".pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='For plotting the training/testing curves')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        help='For choosing which optimizer')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='For choosing a default dataset from torchvision')
    args = parser.parse_args()
    main(args)
