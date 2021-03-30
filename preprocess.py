import torch
from PIL import Image
import numpy as np


def preproc_img(img_path, mask_path):
    """
    Load an image and a mask.

    :param img_path: Path to target image
    :param mask_path: Path to mask image
    :return: The mask, and a 4x256x256 tensor containing the masked RGB image + the binary mask channel
    """
    # open mask and make values binary
    mask = np.array(Image.open(mask_path).convert("L"))
    mask[mask <= 128] = 0
    mask[mask > 128] = 255

    # open image and apply mask by making masked pixels black
    im = np.array(Image.open(img_path).convert("RGB"))
    im[mask == 0, :] = 0

    sample = torch.cat((
        torch.tensor(im),
        torch.tensor(mask.reshape(256, 256, 1))),  # add dimension to match shape of 256x256x3 image
        dim=2)

    # move 'channels' dimension from last to first, as required by PyTorch
    # new shape is 4x256x256
    return mask, torch.movedim(sample, 2, 0).float()