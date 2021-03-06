import argparse
import time

import torch
from fastai.layers import NormType
from fastai.metrics import LossMetrics
from fastai.vision.learner import unet_learner
from torchvision.models import resnet50

from data import UpscalerDataset
from loss import get_loss

lr = 3e-3
wd = 1e-3

bs = 32
size = 128

arch = resnet50


def do_fit(learner, wd, lrs, pct_start=0.9):
    learner.fit_one_cycle(10, lrs, pct_start=pct_start, wd=wd)


def make_meta(arguments) -> dict:
    return {
        'time': str(time.time()),
        'arguments': arguments.__dict__,
        'base_arch': 'resnet50'
    }


def train(arguments):
    dls = UpscalerDataset(arguments.dataset_path, arguments.resized_dataset_path)
    feat_loss = get_loss(arguments.loss_model_path)

    learner = unet_learner(dls.get_dataloaders(bs, size), arch, loss_func=feat_loss,
                           metrics=LossMetrics(feat_loss.metric_names),
                           blur=True, norm_type=NormType.Weight)

    # stage 1
    print('stage 1')
    do_fit(learner, wd, slice(lr * 10))
    learner.unfreeze()
    do_fit(learner, wd, slice(1e-5, lr))

    # checkpoint
    learner.save('checkpoint')
    learner.load('checkpoint')
    # stage 2
    print('stage 2')
    del learner.dls
    learner.dls = dls.get_dataloaders(5, size * 2)
    learner.freeze()
    do_fit(learner, wd, slice(lr))
    learner.unfreeze()
    do_fit(learner, wd, slice(1e-6, 1e-4), pct_start=0.3)

    # save
    torch.save({'model': learner.model, 'meta': make_meta(arguments)}, arguments.output)




def get_arguments():
    parser = argparse.ArgumentParser(description='Train an upscaler')
    parser.add_argument('--dataset-path', help='path to full_size dataset')
    parser.add_argument('--resized-dataset-path', help='path to resized dataset')
    parser.add_argument('--loss-model-path', help='path to loss model')
    parser.add_argument('--output', help='output', default='./super_res.pth')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = get_arguments()
    train(arguments)
