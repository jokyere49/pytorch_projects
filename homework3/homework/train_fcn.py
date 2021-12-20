import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # data Augmentation
    transforms = [dense_transforms.ColorJitter(brightness=0.8, contrast=0.5, saturation=0.8, hue=[-0.5, 0.5]),
                  dense_transforms.RandomHorizontalFlip(flip_prob=0.5),
                  dense_transforms.ToTensor()]
    transform_train = dense_transforms.Compose(transforms=transforms)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = FCN().to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))

    # class weights
    weights = torch.tensor(DENSE_CLASS_DISTRIBUTION, dtype=torch.float32)
    weights = 1.0/weights

    #loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # using  a scheduler for early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    train_path = "dense_data/train"
    valid_path = "dense_data/valid"
    trainDataLoader = load_dense_data(train_path, transform=transform_train, num_workers=0)
    validDataLoader = load_dense_data(valid_path, num_workers=0)

    global_step = 0
    for epoch in range(args.num_epoch):
        # training
        model.train()
        confusion = ConfusionMatrix(5)
        for img, label in trainDataLoader:
            label = label.type(torch.LongTensor)
            img, label = img.to(device), label.to(device)
            logit = model(img)
            loss_val = loss_fn(logit, label)
            confusion.add(logit.argmax(1).cpu(), label.cpu())

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                log(train_logger, img, label, logit, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_acc = confusion.global_accuracy
        avg_iou = confusion.iou
        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)
            train_logger.add_scalar('iou', avg_iou, global_step)

        # model evaluation
        model.eval()
        confusion = ConfusionMatrix(5)
        for img, label in validDataLoader:
            label = label.type(torch.LongTensor)
            img, label = img.to(device), label.to(device)
            confusion.add(model(img.to(device)).argmax(1).cpu(), label.cpu())

            if valid_logger is not None:
                log(valid_logger, img, label, model(img), global_step)

        avg_vacc = confusion.global_accuracy
        avg_viou = confusion.iou
        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)
            valid_logger.add_scalar('iou', avg_viou, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        save_model(model)

        # log the learning rate
        if train_logger:
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # update the scheduler
        scheduler.step(avg_vacc)
        #scheduler.step(avg_viou)
        # save model after one epoch
        save_model(model)

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-log', '--log_dir')
    parser.add_argument('-lr', '--lr', type=float, default=1e-2)
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    # parser.add_argument('-B', '--batch_size', type=int, default=128)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()
    train(args)
