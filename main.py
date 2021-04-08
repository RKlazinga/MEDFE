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
                def to_im_shape(t: torch.Tensor, x: int = 256, y: int = 256):
                    first_in_batch = t.split([1, args.batch_size - 1], dim=0)[0]
                    unmasked = first_in_batch
                    if first_in_batch.shape[1] == 4:
                        unmasked = first_in_batch.split([3, 1], dim=1)[0]
                    return unmasked.reshape(3, x, y)

                im_masked_image = to_im_shape(batch['masked_image'])
                im_gt = to_im_shape(gt256)
                im_gt_smooth = to_im_shape(batch['gt_smooth'], 32, 32)
                if model.struct_branch_img is not None:
                    im_st = to_im_shape(model.struct_branch_img, 32, 32)
                    im_te = to_im_shape(model.tex_branch_img, 32, 32)
                im_out = torch.clamp(out[0], 0, 1)

                im = PIL.Image.new('RGB', (3 * 256, 2 * 256))
                im.paste(to_pil(im_masked_image), (0, 0))
                im.paste(to_pil(im_gt), (256, 0))
                im.paste(CustomDataset.scale(to_pil(im_gt_smooth), 256, resample_method=PIL.Image.NEAREST), (512, 0))
                if model.struct_branch_img is not None:
                    im.paste(CustomDataset.scale(to_pil(im_st), 256, resample_method=PIL.Image.NEAREST), (0, 256))
                    im.paste(CustomDataset.scale(to_pil(im_te), 256, resample_method=PIL.Image.NEAREST), (256, 256))
                im.paste(to_pil(im_out), (512, 256))

                tkimg = PIL.ImageTk.PhotoImage(im)
                iml = tk.Label(win, image=tkimg)
                iml.pack()

                ll = tk.Label(win, text=f"| || || |_: {single_loss.item()}")
                ll.pack()
                win.update()
                iml.pack_forget()
                ll.pack_forget()

        loss /= len(train_loader)

        for k, v in loss_components.items():
            print('\t', k, ' = ', (v / len(train_loader)).item(), sep='')
        torch.save(model.state_dict(), "MODEL")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the image inpainting network')

    parser.add_argument('--train-size', default=100, type=int, help='the number of images to train with')
    parser.add_argument('--batch-size', default=25, type=int, help='the number of images to train with in a single batch')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='the learning rate')
    parser.add_argument('--cuda', action='store_true', help='run with CUDA')
    parser.add_argument('--output-intermediates', action='store_true', help='show intermediate results in a GUI window')

    iargs = parser.parse_args()

    main(iargs)
