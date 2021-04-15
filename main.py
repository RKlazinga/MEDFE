import argparse
import gc

import torch
from torch import optim
from torch.nn.functional import interpolate
from torch.utils import data
import torch.autograd

from training.dataset import CustomDataset
from display.output_renderer import OutputRenderer
from training.loss import TotalLoss
from network.medfe import MEDFE
from training.wgan import Discriminator, train_discriminator


IMG_FOLDER = "data/celeba/img_align_celeba"


def main(args):
    """
    Run a full training of the MEDFE network.
    This requires a dataset to be present in IMG_FOLDER.
    The network also expects smoothed "structure images".

    :param args: Arguments supplied by argparse
    """
    device_name = 'cpu'
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is requested, but not available')
        device_name = 'cuda'

    device = torch.device(device_name)

    smooth_img_folder = IMG_FOLDER + "_tsmooth"
    train_size = args.training_size  # celeba dataset is 202k images large
    training_set = CustomDataset(IMG_FOLDER, smooth_img_folder, train_size)
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # configure the network
    model = MEDFE(
        batch_norm=True,
        use_bpa=False,
        use_branch=True,
        use_res=True,
        branch_channels=512//4,
        channels=64//4
    ).to(device)

    # set up discriminators and their optimisers
    wgan_global = Discriminator((args.batch_size, 3, 256, 256), name='global')
    wgan_local = None
    if training_set.mask_is_rect():
        wgan_local = Discriminator((args.batch_size, 3, training_set.mask[2], training_set.mask[3]), name='local')
    wgan_global_real_hist = None

    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    wgan_global_optimizer = optim.Adam(wgan_global.parameters(), lr=1e-4, betas=(0.5, 0.999))
    wgan_local_optimizer = None
    if wgan_local is not None:
        wgan_local_optimizer = optim.Adam(wgan_local.parameters(), lr=1e-4, betas=(0.5, 0.999))
    wgan_local_real_hist = None

    criterion = TotalLoss(wgan_global, wgan_local)

    renderer = None
    if args.output_intermediates:
        renderer = OutputRenderer(args, model, train_loader)

    for epoch in range(100):
        loss = 0
        loss_components = {}
        for batch_idx, batch in enumerate(train_loader):
            gc.collect()

            optimiser.zero_grad()

            model.set_mask(batch["mask"])
            print(f"Prediction on batch {batch_idx}")

            # get the ground truth image and scale it down
            gt256 = batch["gt"]
            gt32 = interpolate(gt256, 32)

            # feed forward
            out = model(batch["masked_image"])

            # only apply out image to masked area
            mask = model.mask.reshape(model.mask.shape[0], 1, model.mask.shape[1], model.mask.shape[2])
            out = (gt256 * mask + out * (1 - mask)).float()

            # train discriminators
            if wgan_global_real_hist is None:
                wgan_global_real_hist = gt256
            if wgan_global_real_hist.shape[0] > 4 * args.batch_size:
                wgan_global_real_hist = wgan_global_real_hist[:-4 * args.batch_size]
            wgan_global_real_hist = torch.cat((wgan_global_real_hist, gt256), dim=0).detach()
            train_discriminator(wgan_global, wgan_global_optimizer, gt=wgan_global_real_hist, out=out)

            out_sliced = None
            if wgan_local is not None:
                if wgan_local_real_hist is None:
                    wgan_local_real_hist = batch['gt_sliced']
                if wgan_local_real_hist.shape[0] > 4 * args.batch_size:
                    wgan_local_real_hist = wgan_local_real_hist[:-4 * args.batch_size]
                wgan_local_real_hist = wgan_local_real_hist.detach()
                out_sliced = training_set.slice_mask(out.detach())
                train_discriminator(wgan_local, wgan_local_optimizer, gt=wgan_local_real_hist, out=out_sliced)

            # determine the loss
            single_loss = criterion(
                i_gt_small=gt32,
                i_st=batch["gt_smooth"],
                i_ost=model.struct_branch_img,
                i_ote=model.tex_branch_img,
                i_gt_large=gt256,
                i_out_large=out,
                i_gt_sliced=batch['gt_sliced'],
                i_out_sliced=out_sliced,
                mask_size=256*256 - torch.sum(model.mask[0]),
            )

            single_loss.backward()
            optimiser.step()

            loss += single_loss.item()
            for k, v in criterion.last_loss.items():
                loss_components[k] = loss_components.get(k, 0) + v.item()

            if args.output_intermediates:
                renderer.loss_history.append(single_loss.item())
                renderer.loss_components_history.append(
                    {k: v.item() for k, v in criterion.last_loss.items()}
                )

                renderer.update(batch, out, out_sliced)

        loss /= len(train_loader)

        for k, v in loss_components.items():
            print('\t', k, ' = ', (v / len(train_loader)), sep='')
        torch.save(model.state_dict(), "MODEL")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the image inpainting network')

    parser.add_argument('--training-size', default=50000, type=int, help='the number of images to training with')
    parser.add_argument('--batch-size', default=25, type=int, help='the number of images to training with in a single batch')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='the learning rate')
    parser.add_argument('--cuda', action='store_true', help='run with CUDA')
    parser.add_argument('--output-intermediates', action='store_true', help='show intermediate results in a GUI window')

    iargs = parser.parse_args()

    main(iargs)
