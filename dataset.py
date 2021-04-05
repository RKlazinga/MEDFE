import os

import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms


class CustomDataset(data.Dataset):
    def __init__(self, img_dir, smooth_dir, size):
        super().__init__()
        self.img_dir = img_dir
        self.smooth_dir = smooth_dir
        self.images = os.listdir(self.img_dir)[:size]
        self.to_tensor = transforms.ToTensor()

        self.all_1_mask_32 = torch.tensor(np.full((32, 32), 1.0).reshape(1, 32, 32))
        self.all_1_mask_256 = torch.tensor(np.full((256, 256), 1.0).reshape(1, 256, 256))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name = self.images[item]
        img_path = f"{self.img_dir}/{name}"
        smooth_img_path = f"{self.smooth_dir}/{name}"

        unmasked_img = self._add_mask_to_image(self.scale(Image.open(img_path).convert("RGB")), self.all_1_mask_256)
        unmasked_smooth_img = self._add_mask_to_image(
            self.scale(Image.open(smooth_img_path).convert("RGB"), 32),
            self.all_1_mask_32
        )
        mask, masked_img_tensor = self.preproc_img(img_path, None)

        sample = {
            "masked_image": masked_img_tensor,
            "mask": mask,
            "gt": unmasked_img,
            "gt_smooth": unmasked_smooth_img,
        }
        return sample

    def _add_mask_to_image(self, img, mask):
        return torch.cat((self.to_tensor(img) / 255.0, mask), dim=0).float()

    @staticmethod
    def scale(im: Image.Image, size=256):
        # reshape im to be square
        if im.height > im.width:
            im = im.crop((0, int((im.height - im.width)/2), im.width, im.height - int((im.height - im.width)/2)))
        elif im.width > im.height:
            im = im.crop((int((im.width - im.height)/2), 0, im.width - int((im.width - im.height)/2), im.height))

        return im.resize((size, size), resample=Image.BILINEAR)

    @staticmethod
    def preproc_img(img_path, mask_path=None):
        """
        Load an image and a mask.

        :param img_path: Path to target image
        :param mask_path: Path to mask image. If None, a center hole mask will be used.
        :return: The mask, and a 4x256x256 tensor containing the masked RGB image + the binary mask channel
        """
        if mask_path is not None:
            # open mask and make values binary
            mask = np.array(Image.open(mask_path).convert("L")) / 255.0
            mask[mask <= 0.5] = 0
            mask[mask > 0.5] = 1
        else:
            mask = np.full((256, 256), 1)
            mask[64:-64, 64:-64] = 0

        # open image and apply mask by making masked pixels black
        im = Image.open(img_path).convert("RGB")
        im = np.array(CustomDataset.scale(im)) / 255.0
        im[mask == 0, :] = 0

        sample = torch.cat((
            torch.tensor(im),
            torch.tensor(mask.reshape(256, 256, 1))),  # add dimension to match shape of 256x256x3 image
            dim=2)

        # move 'channels' dimension from last to first, as required by PyTorch
        # new shape is 4x256x256
        return mask, torch.movedim(sample, 2, 0).float()
