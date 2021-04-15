import torch

from network.medfe import MEDFE
from torch.utils import data
from typing import List, Dict, Optional
import tkinter as tk
from torchvision import transforms
import PIL.Image
import PIL.ImageTk

from training.dataset import CustomDataset


class OutputRenderer:
    """
    Display training progress using a tkinter window.
    Visualises various intermediate model images and the loss over time.
    """
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

    def update(self, batch, out, out_sliced: Optional):
        def to_im_shape(t: torch.Tensor, x: int = 256, y: int = 256):
            first_in_batch = t.split([1, self.args.batch_size - 1], dim=0)[0]
            unmasked = first_in_batch
            if first_in_batch.shape[1] == 4:
                unmasked = first_in_batch.split([3, 1], dim=1)[0]
            return torch.clamp(unmasked.reshape(3, x, y), 0, 1)

        im_masked_image = to_im_shape(batch['masked_image'])
        im_gt = to_im_shape(batch['gt'])
        im_st = im_te = None
        if self.model.struct_branch_img is not None:
            im_st = to_im_shape(self.model.struct_branch_img, 32, 32)
            im_te = to_im_shape(self.model.tex_branch_img, 32, 32)
        im_out = to_im_shape(out)
        im_gt_sliced = im_out_sliced = im_gt_smooth = None
        if out_sliced is not None:
            slice_shape = batch['gt_sliced'].shape
            im_gt_sliced = to_im_shape(batch['gt_sliced'], slice_shape[2], slice_shape[3])
            im_out_sliced = to_im_shape(out_sliced, slice_shape[2], slice_shape[3])
        else:
            im_gt_smooth = to_im_shape(batch['gt_smooth'], 32, 32)

        im = PIL.Image.new('RGB', (3 * 256, 2 * 256))
        im.paste(self.to_pil(im_masked_image), (0, 0))
        im.paste(self.to_pil(im_gt), (256, 0))
        if out_sliced is not None:
            im.paste(self.to_pil(im_gt_sliced), (512, 0))
            im.paste(self.to_pil(im_out_sliced), (512 + 128, 0))
        else:
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
