import torch
import numpy as np
import torch.nn.functional as F
from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
from .utils import PR
from .utils import DetectionSuperTuxDataset


# Helper functions

def point_in_box(pred, lbl):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return (x0 <= px) & (px < x1) & (y0 <= py) & (py < y1)


def point_close(pred, lbl, d=5):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return ((x0 + x1 - 1) / 2 - px) ** 2 + ((y0 + y1 - 1) / 2 - py) ** 2 < d ** 2


def box_iou(pred, lbl, t=0.5):
    px, py, pw2, ph2 = pred[:, None, 0], pred[:, None, 1], pred[:, None, 2], pred[:, None, 3]
    px0, px1, py0, py1 = px - pw2, px + pw2, py - ph2, py + ph2
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    iou = (abs(torch.min(px1, x1) - torch.max(px0, x0)) * abs(torch.min(py1, y1) - torch.max(py0, y0))) / \
          (abs(torch.max(px1, x1) - torch.min(px0, x0)) * abs(torch.max(py1, y1) - torch.min(py0, y0)))
    return iou > t


class Focalloss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-BCE_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)
        F_loss = alpha_tensor * (1 - p_t) ** self.gamma * BCE_loss
        return F_loss.mean()


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # data Augmentation
    transforms = [dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
                  dense_transforms.RandomHorizontalFlip(flip_prob=0.5),
                  dense_transforms.ToTensor(),
                  dense_transforms.ToHeatmap()]
    val_transforms = [dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]
    transform_train = dense_transforms.Compose(transforms=transforms)
    transform_val = dense_transforms.Compose(transforms=val_transforms)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    loss_fn = Focalloss(alpha=0.9, gamma=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # using  a scheduler for early stopping
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    train_path = "dense_data/train"
    valid_path = "dense_data/valid"
    trainDataLoader = load_detection_data(train_path, transform=transform_train, num_workers=0)
    validDataLoader = load_detection_data(valid_path, transform=transform_val, num_workers=0)

    global_step = 0
    for epoch in range(args.num_epoch):
        # training
        model.train()

        for img, label, _ in trainDataLoader:

            img, label = img.to(device), label.to(device)
            logit = model(img)
            loss_val = loss_fn(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if train_logger:
            log(train_logger, img, label, logit, global_step)

        # model evaluation
        model.eval()
        pr_box = [PR() for _ in range(3)]
        pr_dist = [PR(is_close=point_close) for _ in range(3)]
        pr_iou = [PR(is_close=box_iou) for _ in range(3)]

        # show a picture for validation
        for img, label, _ in validDataLoader:
            img, label = img.to(device), label.to(device)
            if valid_logger is not None:
                log(valid_logger, img, label, model(img), global_step)
            break

        # evaluation

        for img, *gts in DetectionSuperTuxDataset('dense_data/valid', min_size=0):
            with torch.no_grad():
                detections = model.detect(img.to(device))
                for i, gt in enumerate(gts):
                    pr_box[i].add(detections[i], gt)
                    pr_dist[i].add(detections[i], gt)
                    pr_iou[i].add(detections[i], gt)
        print('ap_c0:', pr_box[0].average_prec)
        print('ap_c1:', pr_box[1].average_prec)
        print('ap_c2:', pr_box[1].average_prec)
        if valid_logger is not None:
            valid_logger.add_scalar('ap_c0', pr_box[0].average_prec, global_step)
            valid_logger.add_scalar('ap_c1', pr_box[1].average_prec, global_step)
            valid_logger.add_scalar('ap_c2', pr_box[2].average_prec, global_step)

        # log the learning rate
        if train_logger:
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        print("{}/{} completed.............!!!".format(epoch + 1, args.num_epoch))
        save_model(model)

    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-log', '--log_dir')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-n', '--num_epoch', type=int, default=100)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()
    train(args)
