import argparse

import torch
from torch import nn, optim
from torch.nn.functional import interpolate
from torch.utils import tensorboard, data

from loss import TotalLoss
from network import MEDFE
from dataset import CustomDataset


def main(args):
    device_name = 'cpu'
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is requested, but not available')
        device_name = 'cuda'

    device = torch.device(device_name)

    img_folder = "data/celeba/img_align_celeba"
    train_size = args.train_size  # celeba dataset is 202k images large
    training_set = CustomDataset(img_folder, img_folder+"_tsmooth", train_size)
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    model = MEDFE().to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = TotalLoss(model)

    for epoch in range(10):
        loss = 0
        for batch_idx, batch in enumerate(train_loader):
            model.set_mask(batch["mask"])
            print(f"Prediction on batch {batch_idx}")
            out = model(batch["masked_image"])
            out32 = interpolate(out, 32)

            gt256 = batch["gt"]
            gt32 = interpolate(gt256, 32)

            single_loss = criterion(
                gt32,
                batch["gt_smooth"],
                model.struct_branch_img,
                model.tex_branch_img,
                out32,
                gt256,
                out
            )
            single_loss.backward()
            optimiser.step()
            loss += single_loss.item()
        loss /= len(train_loader)
        print(epoch, loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the image inpainting network')

    parser.add_argument('--train-size', default=10, type=int, help='the number of images to train with')
    parser.add_argument('--batch-size', default=1, type=int, help='the number of images to train with in a single batch')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='the learning rate')
    parser.add_argument('--cuda', action='store_true', help='run with CUDA')

    main(parser.parse_args())
