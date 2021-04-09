import argparse
import tkinter as tk
import gc
from typing import List, Dict

import torch
from torch import optim
from torch.nn.functional import interpolate
from torch.utils import data
from torchvision import transforms

from dataset import CustomDataset
from loss import TotalLoss
from network.network import MEDFE


class OutputRenderer:
    def __init__(self, args, model: MEDFE, train_loader: data.DataLoader):
        import matplotlib.figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        matplotlib.use('TkAgg')

        self.args = args
        self.model = model
        self.train_loader = train_loader

        self.loss_history: List[float] = []
        self.loss_components_history: List[Dict] = []

        self.to_pil = transforms.ToPILImage()

        self.win = tk.Tk()

        self.fig = matplotlib.figure.Figure(figsize=(8, 5))
        self.loss_ax = self.fig.add_subplot(111)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, self.win)
        self.fig_canvas.get_tk_widget().pack(side=tk.RIGHT)

        self.collage = tk.Label(self.win)
        self.collage.pack()

    def update(self, batch, out):
        import PIL.Image
        import PIL.ImageTk

        def to_im_shape(t: torch.Tensor, x: int = 256, y: int = 256):
            first_in_batch = t.split([1, self.args.batch_size - 1], dim=0)[0]
            unmasked = first_in_batch
            if first_in_batch.shape[1] == 4:
                unmasked = first_in_batch.split([3, 1], dim=1)[0]
            return torch.clamp(unmasked.reshape(3, x, y), 0, 1)

        im_masked_image = to_im_shape(batch['masked_image'])
        im_gt = to_im_shape(batch['gt'])
        im_gt_smooth = to_im_shape(batch['gt_smooth'], 32, 32)
        im_st = im_te = None
        if self.model.struct_branch_img is not None:
            im_st = to_im_shape(self.model.struct_branch_img, 32, 32)
            im_te = to_im_shape(self.model.tex_branch_img, 32, 32)
        im_out = to_im_shape(out)

        im = PIL.Image.new('RGB', (3 * 256, 2 * 256))
        im.paste(self.to_pil(im_masked_image), (0, 0))
        im.paste(self.to_pil(im_gt), (256, 0))
        im.paste(CustomDataset.scale(self.to_pil(im_gt_smooth), 256, resample_method=PIL.Image.NEAREST), (512, 0))
        if self.model.struct_branch_img is not None:
            im.paste(CustomDataset.scale(self.to_pil(im_st), 256, resample_method=PIL.Image.NEAREST), (0, 256))
            im.paste(CustomDataset.scale(self.to_pil(im_te), 256, resample_method=PIL.Image.NEAREST), (256, 256))
        im.paste(self.to_pil(im_out), (512, 256))

        tkimg = PIL.ImageTk.PhotoImage(im)
        self.collage.configure(image=tkimg)
        self.collage.image = self.collage

        self.loss_ax.clear()
        history_x = [i / len(self.train_loader) for i in range(0, len(self.loss_history))]
        for loss_type in self.loss_components_history[0].keys():
            self.loss_ax.plot(
                history_x, [l[loss_type] for l in self.loss_components_history],
                label=loss_type,
            )
        self.loss_ax.plot(history_x, self.loss_history, 'k', linewidth=2, label='total')
        self.loss_ax.legend(loc='upper right')
        self.fig_canvas.draw()

        self.win.update()


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
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = MEDFE(branch_channels=64,
                  channels=8).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = TotalLoss()

    renderer = None
    if args.output_intermediates:
        renderer = OutputRenderer(args, model, train_loader)

    for epoch in range(1000):
        loss = 0
        loss_components = {}
        for batch_idx, batch in enumerate(train_loader):
            gc.collect()

            optimiser.zero_grad()
            model.set_mask(batch["mask"])
            print(f"Prediction on batch {batch_idx}")

            gt256 = batch["gt"]
            gt32 = interpolate(gt256, 32)
            out = model(batch["masked_image"])

            # only apply out image to masked area
            print(gt256.shape)
            print(model.mask.shape)
            mask = model.mask.reshape(model.mask.shape[0], 1, model.mask.shape[1], model.mask.shape[2])
            out = gt256 * mask + out * (1 - mask)

            single_loss = criterion(
                gt32,
                batch["gt_smooth"],
                model.struct_branch_img,
                model.tex_branch_img,
                gt256,
                out,
                256*256 - torch.sum(model.mask[0])
            )
            single_loss.backward()
            optimiser.step()
            loss += single_loss.item()
            for k, v in criterion.last_loss.items():
                loss_components[k] = loss_components.get(k, 0) + v

            if args.output_intermediates:
                renderer.loss_history.append(single_loss.item())
                renderer.loss_components_history.append(
                    {k: v.item() for k, v in criterion.last_loss.items()}
                )

                renderer.update(batch, out)

        loss /= len(train_loader)

        for k, v in loss_components.items():
            print('\t', k, ' = ', (v / len(train_loader)).item(), sep='')
        torch.save(model.state_dict(), "MODEL")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the image inpainting network')

    parser.add_argument('--train-size', default=5000, type=int, help='the number of images to train with')
    parser.add_argument('--batch-size', default=50, type=int, help='the number of images to train with in a single batch')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='the learning rate')
    parser.add_argument('--cuda', action='store_true', help='run with CUDA')
    parser.add_argument('--output-intermediates', action='store_true', help='show intermediate results in a GUI window')

    iargs = parser.parse_args()

    main(iargs)
