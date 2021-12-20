from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # data Augmentation
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.8, hue=[-0.5, 0.5]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
    ])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNNClassifier().to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # using  a scheduler for early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    train_path = "data/train"
    valid_path = "data/valid"
    trainDataLoader = load_data(train_path, transform=transform_train, num_workers=0, batch_size=args.batch_size)
    validDataLoader = load_data(valid_path, num_workers=0, batch_size=args.batch_size)

    global_step = 0
    for epoch in range(args.num_epoch):
        # training
        model.train()
        confusion = ConfusionMatrix(6)
        for img, label in trainDataLoader:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss_fn(logit, label)
            confusion.add(logit.argmax(1).cpu(), label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_acc = confusion.global_accuracy
        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        # model evaluation
        model.eval()
        confusion = ConfusionMatrix(6)
        for img, label in validDataLoader:
            img, label = img.to(device), label.to(device)
            confusion.add(model(img.to(device)).argmax(1).cpu(), label)

        avg_vacc = confusion.global_accuracy

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        save_model(model)

        # log the learning rate
        if train_logger:
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # update the scheduler
        scheduler.step(avg_vacc)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-log', '--log_dir')
    parser.add_argument('-lr', '--lr', type=float, default=1e-2)
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-B', '--batch_size', type=int, default=128)
    parser.add_argument('-c', '--continue_training', action='store_true')
    # parser.add_argument('-c', '--continue_training', action='store_false')
    args = parser.parse_args()
    train(args)
