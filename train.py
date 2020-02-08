import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from dataloader import UnfoldIcoDataset, ToTensor, Normalize
from models.hexrunet import HexRUNet_C

parser = argparse.ArgumentParser(description='Training for OmniMNIST',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epochs', default=30, type=int, metavar='N', help='total epochs')
parser.add_argument('--pretrained', default=None, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--level', default=4, type=int, metavar='N', help='max level for icosahedron')
parser.add_argument('-b', '--batch-size', default=15, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--arch', default='hexrunet', type=str, help='architecture name for log folder')
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='tensorboard log interval')
parser.add_argument('--train_rot', action='store_true', help='rotate image for trainset')
parser.add_argument('--test_rot', action='store_true', help='rotate image for testset')


def main():
    args = parser.parse_args()
    print('Arguments:')
    print(json.dumps(vars(args), indent=1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True
    print("device:", device)

    print('=> setting data loader')
    erp_shape = (60, 120)
    transform = transforms.Compose([ToTensor(), Normalize((0.0645,), (0.2116,))])
    trainset = UnfoldIcoDataset(datasets.MNIST(root='raw_data', train=True, download=True),
                                erp_shape, args.level, rotate=args.train_rot, transform=transform)
    testset = UnfoldIcoDataset(datasets.MNIST(root='raw_data', train=False, download=True),
                               erp_shape, args.level, rotate=args.test_rot, transform=transform)
    train_loader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(testset, args.batch_size, shuffle=False, num_workers=args.workers)

    print('=> setting model')
    start_epoch = 0
    model = HexRUNet_C(1)
    total_params = 0
    for param in model.parameters():
        total_params += np.prod(param.shape)
    print(f"Total model parameters: {total_params:,}.")
    model = model.to(device)

    # Loss function
    print('=> setting loss function')
    criterion = nn.CrossEntropyLoss()

    # setup solver scheduler
    print('=> setting optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('=> setting scheduler')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        print("=> using pre-trained weights")
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Resume training from epoch {}".format(start_epoch))

    timestamp = datetime.now().strftime("%m%d-%H%M")
    log_folder = join('checkpoints', f'{args.arch}_{timestamp}')
    print(f'=> create log folder: {log_folder}')
    os.makedirs(log_folder, exist_ok=True)
    with open(join(log_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    writer = SummaryWriter(log_dir=log_folder)
    writer.add_text('args', json.dumps(vars(args), indent=1))

    # Training
    for epoch in range(start_epoch, args.epochs):

        # --------------------------
        # training
        # --------------------------
        model.train()
        losses = []
        pbar = tqdm(train_loader)
        total = 0
        correct = 0
        mode = 'train'
        for idx, batch in enumerate(pbar):
            # to cuda
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            outputs = model(batch)
            labels = batch['label']

            # Loss function
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # update progress bar
            display = OrderedDict(mode=f'{mode}', epoch=f"{epoch:>2}", loss=f"{losses[-1]:.4f}")
            pbar.set_postfix(display)

            # tensorboard log
            if idx % args.log_interval == 0:
                niter = epoch * len(train_loader) + idx
                writer.add_scalar(f'{mode}/loss', loss.item(), niter)

        # End of one epoch
        scheduler.step()
        ave_loss = sum(losses) / len(losses)
        ave_acc = 100 * correct / total
        writer.add_scalar(f'{mode}/loss_ave', ave_loss, epoch)
        writer.add_scalar(f'{mode}/acc_ave', ave_acc, epoch)

        print(f"Epoch:{epoch}, Train Loss average:{ave_loss:.4f}, Accuracy average:{ave_acc:.2f}")

        save_data = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ave_loss': ave_loss,
        }
        torch.save(save_data, join(log_folder, f'checkpoints_{epoch}.pth'))

        # --------------------------
        # evaluation
        # --------------------------
        model.eval()
        losses = []
        pbar = tqdm(test_loader)
        total = 0
        correct = 0
        mode = 'test'
        for idx, batch in enumerate(pbar):
            with torch.no_grad():
                # to cuda
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                outputs = model(batch)
                labels = batch['label']

                # Loss function
                loss = criterion(outputs, labels)
                losses.append(loss.item())

            # accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # update progress bar
            display = OrderedDict(mode=f'{mode}', epoch=f"{epoch:>2}", loss=f"{losses[-1]:.4f}")
            pbar.set_postfix(display)

            # tensorboard log
            if idx % args.log_interval == 0:
                niter = epoch * len(test_loader) + idx
                writer.add_scalar(f'{mode}/loss', loss.item(), niter)

        # End of one epoch
        ave_loss = sum(losses) / len(losses)
        ave_acc = 100 * correct / total
        writer.add_scalar(f'{mode}/loss_ave', ave_loss, epoch)
        writer.add_scalar(f'{mode}/acc_ave', ave_acc, epoch)

        print(f"Epoch:{epoch}, Test Loss average:{ave_loss:.4f}, Accuracy average:{ave_acc:.2f}")


if __name__ == '__main__':
    main()
