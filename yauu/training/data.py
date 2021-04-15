from typing import Callable

from fastai.vision.all import *
from pathlib import Path

import torch


class UpscalerDataset:
    def __init__(self, hr_path: Path, lr_path: Path):
        self.hr_path = hr_path
        self.lr_path = lr_path

    def _get_y(self, file_path):
        return self.hr_path / file_path.parent.relative_to(self.lr_path) / file_path.name

    def get_dataloaders(self, batch_size, size):
        dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                           get_items=get_image_files,
                           get_y=lambda x: self._get_y(x),
                           splitter=RandomSplitter(),
                           item_tfms=Resize(size),
                           batch_tfms=[*aug_transforms(), Normalize()])
        dblock.summary(self.lr_path)
        dls = dblock.dataloaders(self.lr_path, bs=batch_size)
        dls.c = 3
        return dls

