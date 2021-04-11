import os
from collections import namedtuple
from typing import Union

import torch
from torch.nn import L1Loss
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

# x, y, w, h
Rect = namedtuple('Rect', 'x y w h')


class CustomDataset(data.Dataset):
    def __init__(self, img_dir, smooth_dir, size, mask: Union[Rect, str] = Rect(64, 64, 128, 128)):
        super().__init__()
        self.img_dir = img_dir
        self.smooth_dir = smooth_dir
        self.images = os.listdir(self.img_dir)[:size]
        self.to_tensor = transforms.ToTensor()
        self.mask = mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        """
        Fetch and prepare a single image

        :param item: Filename of the image
        :return: A dictionary with the required tensors:
         - 'masked_image': 4x256x256 tensor containing image with mask
         - 'mask':  256x256x1 tensor containing the location of the mask
         - 'gt': 3x256x256 tensor containing the unmasked image
         - 'gt_smooth': 3x32x32 tensor containing a smoothed representation of the image
        """
        name = self.images[item]
        img_path = f"{self.img_dir}/{name}"
        smooth_img_path = f"{self.smooth_dir}/{name}"

        unmasked_img = self.to_tensor(self.scale(Image.open(img_path).convert("RGB"), 256))
        unmasked_smooth_img = self.to_tensor(self.scale(Image.open(smooth_img_path).convert("RGB"), 32))

        mask, masked_img_tensor = self.preproc_img(img_path, self.mask)

        sample = {
            "masked_image": masked_img_tensor,
            "mask": mask,
            "gt": unmasked_img,
            "gt_smooth": unmasked_smooth_img,
        }

        if self.mask_is_rect():
            sample['gt_sliced'] = self.slice_mask(unmasked_img)

        for t in sample.values():
            t.requires_grad = False

        return sample

    @staticmethod
    def scale(im: Image.Image, size, resample_method=Image.BILINEAR):
        # reshape im to be square
        if im.height > im.width:
            im = im.crop((0, int((im.height - im.width)/2), im.width, im.height - int((im.height - im.width)/2)))
        elif im.width > im.height:
            im = im.crop((int((im.width - im.height)/2), 0, im.width - int((im.width - im.height)/2), im.height))

        return im.resize((size, size), resample=resample_method)

    def preproc_img(self, img_path, mask_spec=None):
        """
        Load an image and a mask.

        :param img_path: Path to target image
        :param mask_spec: Path to mask image or region to mask
        :return: The mask, and a 4x256x256 tensor containing the masked RGB image + the binary mask channel
        """
        if not self.mask_is_rect():
            # open mask and make values binary
            mask = np.array(Image.open(mask_spec).convert("L")) / 255
            mask[mask <= 0.5] = 0.0
            mask[mask > 0.5] = 1.0
        else:
            mask = np.full((256, 256), 1.0)
            mask[mask_spec[0]:mask_spec[0] + mask_spec[2], mask_spec[1]:mask_spec[1] + mask_spec[3]] = 0.0

        # open image and apply mask by making masked pixels black
        im = Image.open(img_path).convert("RGB")
        im = np.array(CustomDataset.scale(im, 256)) / 255
        im[mask == 0.0, :] = 0.0

        sample = torch.cat((
            torch.tensor(im),
            torch.tensor(mask.reshape(256, 256, 1))),  # add dimension to match shape of 256x256x3 image
            dim=2)

        # move 'channels' dimension from last to first, as required by PyTorch
        # new shape is 4x256x256
        return torch.Tensor(mask), torch.movedim(sample, 2, 0).float()

    def mask_is_rect(self) -> bool:
        return type(self.mask) == Rect

    def slice_mask(self, t):
        if len(t.shape) == 4:
            return t[:, :, self.mask[0]:self.mask[0] + self.mask[2], self.mask[1]:self.mask[1] + self.mask[3]]
        if len(t.shape) == 3:
            return t[:, self.mask[0]:self.mask[0] + self.mask[2], self.mask[1]:self.mask[1] + self.mask[3]]
        if len(t.shape) == 2:
            return t[self.mask[0]:self.mask[0] + self.mask[2], self.mask[1]:self.mask[1] + self.mask[3]]
        raise ValueError(f"Can't slice a {len(t.shape)}D tensor")
