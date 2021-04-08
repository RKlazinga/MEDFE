import argparse

import torch
from torch import nn, optim
from torch.nn.functional import interpolate
from torch.utils import tensorboard, data

from loss import TotalLoss
from network import MEDFE
from dataset import CustomDataset
from torchvision import transforms
import PIL
import PIL.Image
import PIL.ImageTk
import tkinter as tk


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
    criterion = TotalLoss()

    if args.output_intermediates:
        to_pil = transforms.ToPILImage()
        win = tk.Tk()

    for epoch in range(1000):
        loss = 0
        loss_components = {}
        for batch_idx, batch in enumerate(train_loader):
            optimiser.zero_grad()
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
            for k, v in criterion.last_loss.items():
                loss_components[k] = loss_components.get(k, 0) + v

            if args.output_intermediates:
                im_masked_image = batch['masked_image'].split([3, 1], dim=1)[0].reshape(3, 256, 256)
                im_gt = gt256.reshape(3, 256, 256)
                im_gt_smooth = batch['gt_smooth'].reshape(3, 32, 32)
                im_st = model.struct_branch_img.reshape(3, 32, 32)
                im_te = model.tex_branch_img.reshape(3, 32, 32)
                im_out = out.reshape(3, 256, 256)

                im = PIL.Image.new('RGB', (3 * 256, 2 * 256))
                im.paste(to_pil(im_masked_image), (0, 0))
                im.paste(to_pil(im_gt), (256, 0))
                im.paste(CustomDataset.scale(to_pil(im_gt_smooth), 256, resample_method=PIL.Image.NEAREST), (512, 0))
                im.paste(CustomDataset.scale(to_pil(im_st), 256, resample_method=PIL.Image.NEAREST), (0, 256))
                im.paste(CustomDataset.scale(to_pil(im_te), 256, resample_method=PIL.Image.NEAREST), (256, 256))
                im.paste(to_pil(im_out), (512, 256))
                tkimg = PIL.ImageTk.PhotoImage(im)
                iml = tk.Label(win, image=tkimg)
                iml.pack()
                win.update()
                iml.pack_forget()
        loss /= len(train_loader)

        for k, v in loss_components.items():
            print('\t', k, ' = ', (v / len(train_loader)).item(), sep='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the image inpainting network')

    parser.add_argument('--train-size', default=1, type=int, help='the number of images to train with')
    parser.add_argument('--batch-size', default=1, type=int, help='the number of images to train with in a single batch')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='the learning rate')
    parser.add_argument('--cuda', action='store_true', help='run with CUDA')
    parser.add_argument('--output-intermediates', action='store_true', help='show intermediate results in a GUI window')

    iargs = parser.parse_args()
    if iargs.output_intermediates and iargs.batch_size != 1:
        raise ValueError("Intermediates can only be shown if the batch size is 1")

    main(iargs)
