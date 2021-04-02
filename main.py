import os

from PIL import Image
from torch import nn, optim
from torch.nn.functional import interpolate
from torch.utils import tensorboard, data
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

from loss import TotalLoss
from network import MEDFE
from dataset import CustomDataset


def main():
    img_folder = "data/celeba/img_align_celeba"
    train_size = 10  # celeba dataset is 202k images large
    training_set = CustomDataset(img_folder, img_folder+"_tsmooth", train_size)
    train_loader = data.DataLoader(training_set, batch_size=1, shuffle=True, num_workers=1)

    model = MEDFE()
    optimiser = optim.Adam(model.parameters())
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
    main()
