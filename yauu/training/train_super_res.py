from pathlib import Path

import torch
from fastai.layers import NormType
from fastai.metrics import LossMetrics
from fastai.vision.learner import unet_learner
from torchvision.models import resnet34

from data import UpscalerDataset
from loss import get_loss
import argparse



DATASET_PATH = Path('/home/lleonard/Documents/datasets/best_art/images/512px/')
RESIZED_PATH = Path('/home/lleonard/Documents/datasets/best_art/images/96px/')
LOSS_MODEL_PATH = Path('/home/lleonard/dev/perso/super_res/yaau/nb/painting/models/paintings_artist_classifier.pth')

lr = 3e-3
wd = 1e-3

bs = 32
size = 128

arch = resnet34


def do_fit(learner, wd, lrs, pct_start=0.9):
    learner.fit_one_cycle(10, lrs, pct_start=pct_start, wd=wd)


def main(arguments):
    dls = UpscalerDataset(arguments.dataset_path, arguments.resized_dataset_path)
    feat_loss = get_loss(arguments.loss_model_path)

    learner = unet_learner(dls.get_dataloaders(bs, size), arch, loss_func=feat_loss,
                           metrics=LossMetrics(feat_loss.metric_names),
                           blur=True, norm_type=NormType.Weight)

    # stage 1
    do_fit(learner, wd, slice(lr * 10))
    learner.unfreeze()
    do_fit(learner, wd, slice(1e-5, lr))

    # stage 2
    learner.dls = dls.get_dataloaders(12, size * 2)
    learner.freeze()
    do_fit(learner, wd, slice(lr))
    learner.unfreeze()
    do_fit(learner, wd, slice(1e-6, 1e-4), pct_start=0.3)

    # save
    torch.save({'model': learner.model}, './super_res.pth')


def get_arguments():
    parser = argparse.ArgumentParser(description='Train an upscaler')
    parser.add_argument('--dataset-path', help='path to full_size dataset')
    parser.add_argument('--resized-dataset-path', help='path to resized dataset')
    parser.add_argument('--loss-model-path', help='path to loss model')
    return parser.parse_args()

if __name__ == '__main__':
    arguments = get_arguments()
    main(arguments)
