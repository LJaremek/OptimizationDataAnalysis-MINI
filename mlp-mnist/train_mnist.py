"""
An example script for training a model for the MNIST dataset with BITorch.

Modified from the `PyTorch MNIST Example <https://github.com/pytorch/examples/blob/main/mnist/main.py>`_,
which was published under the `BSD 3-Clause License <https://github.com/pytorch/examples/blob/main/LICENSE>`_.
"""

import os
import time
import pickle
from math import floor
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_mnist import MLP, BinarizedMLP
from model_brevitas_mnist import MLPQuantized
from mnist_tools import train, test
from recorder import plot_results, setup_logging
from parser_args import parse_arguments
import torch.quantization as quant


if __name__ == "__main__":
    args = parse_arguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    # Dataset configuration
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    
    if args.model == 'bimlp':
        model = BinarizedMLP().to(device)
    elif args.model == 'qmlp':
        model = MLPQuantized().to(device)
    else:
        model = MLP().to(device)
    
    # Optimizer selection
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"{args.model}_{args.optimizer}_{timestamp}.txt"
    logger = setup_logging(log_filename, './temp')

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    start_time = time.time()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch, args.log_interval, logger)
        test_loss, test_accuracy = test(model, device, test_loader, logger)
        scheduler.step()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # Time measuring
    end_time = time.time()
    training_time = end_time - start_time

    log_message = f"Total training time: {floor(training_time//60)} min {round(training_time%60, 2)} seconds"
    logger.info(log_message)
    print(log_message)
    
    # Save the model
    if args.save_model:
        model_dir = './checkpoints'
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"{args.model}_{args.optimizer}_{timestamp}.pt"
        model_path = os.path.join(model_dir, model_name)
        torch.save(model.state_dict(), model_path)

    # Plot results
    plot_dir = './plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = f"{args.model}_{args.optimizer}_{timestamp}.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    epochs = range(1, args.epochs + 1)
    plot_results(epochs, test_losses, train_accuracies, test_accuracies, plot_path)


    # Save Accuracies for later comparing plots
    acc_dir = './accuracies'
    os.makedirs(model_dir, exist_ok=True)
    train_acc_filename = f'{args.model}_{args.optimizer}_{timestamp}_train.pickle'
    test_acc_filename = f'{args.model}_{args.optimizer}_{timestamp}_test.pickle'
    
    plot_path = os.path.join(plot_dir, plot_filename)
    
    with open(os.path.join(acc_dir, train_acc_filename), 'wb') as handle:
        pickle.dump(train_accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(acc_dir, test_acc_filename), 'wb') as handle:
        pickle.dump(test_accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)
