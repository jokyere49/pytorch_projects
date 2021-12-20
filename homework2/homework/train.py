from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import numpy as np


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_path = "data/train"
    valid_path = "data/valid"
    trainDataLoader = load_data(train_path, num_workers=0, batch_size=args.batch_size)
    validDataLoader = load_data(valid_path, num_workers=0, batch_size=args.batch_size)
    epochs = args.epochs
    # print(args.batch_size)
    # global_step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # Training
        train_accuracy = []
        train_loss = 0
        size = len(trainDataLoader.dataset)
        num_batches = len(trainDataLoader)
        model.train()
        for batch, (X, y) in enumerate(trainDataLoader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()
            train_accuracy.append(accuracy(pred, y))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss /= num_batches

        # logging to Tensorboard
        train_logger.add_scalar('loss', train_loss, global_step=epoch)
        train_logger.add_scalar('Accuracy', np.mean(train_accuracy), global_step=epoch)

        # printing the training loss and accuracy
        print("train_loss:", train_loss)
        print("train_accuracy:", np.mean(train_accuracy))

        # validation
        test_accuracy = []
        size = len(validDataLoader.dataset)
        num_batches = len(validDataLoader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in validDataLoader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                test_accuracy.append(accuracy(pred, y))
        test_loss /= num_batches

        # logging to Tensorboard
        valid_logger.add_scalar('loss', test_loss, global_step=epoch)
        valid_logger.add_scalar('Accuracy', np.mean(test_accuracy), global_step=epoch)

        # printing the training loss and accuracy
        print("test_loss:", test_loss)
        print("test_accuracy:", np.mean(test_accuracy))
    print("Done !")
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-log', '--log_dir')
    parser.add_argument('-lr', '--lr', type=float, default=1e-2)
    parser.add_argument('-ep', '--epochs', type=int, default=5)
    parser.add_argument('-B', '--batch_size', type=int, default=128)
    args = parser.parse_args()
    train(args)
