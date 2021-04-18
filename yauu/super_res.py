import argparse
import sys
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import numpy as np
import torch
import onnxruntime

from PIL import ImageDraw
from torchvision.transforms import ToTensor


def grid(image: PIL.Image.Image, grid_size, overlap) -> List[List[PIL.Image.Image]]:
    the_grid = []
    w, h = image.size
    for x in range(0, w, grid_size):
        row = []
        for y in range(0, h, grid_size):
            row.append(image.crop(
                (x - overlap // 2, y - overlap // 2, x + grid_size + overlap // 2, y + grid_size + overlap // 2)))
        the_grid.append(row)
    return the_grid


def to_image(x):
    "Convert a tensor or array to a PIL int8 Image"
    if isinstance(x, PIL.Image.Image): return x
    if isinstance(x, torch.Tensor): x = x.permute((1, 2, 0)).cpu().numpy()
    if x.dtype == np.float32: x = (x * 255).astype(np.uint8)
    return PIL.Image.fromarray(x, mode=['RGB', 'CMYK'][x.shape[0] == 4])


class Upscaler:
    def __init__(self, image_path: Path, model, scale: int, grid_size: int, overlap_factor: int, device: str):
        # the image to upscale
        self.image = PIL.Image.open(image_path)
        # the model to use
        self.model = onnxruntime.InferenceSession(model)
        # the scaling factor
        self.scale = scale
        # the base size of each tile
        self.grid_size = grid_size
        # how much to overlap between tiles ?
        self.overlap = self.grid_size // overlap_factor
        # cpu or cuda
        self.device = device
        self.tiles = self._grid()
        # the size of each upscaled tile
        self.big_grid_size = (grid_size + self.overlap) * scale

        self.alpha_circle = PIL.Image.new('1', (self.big_grid_size, self.big_grid_size))
        ImageDraw.Draw(self.alpha_circle).ellipse([(0, 0), (self.big_grid_size, self.big_grid_size)], fill=(255,))

        full_w = (self.big_grid_size - (self.overlap * self.scale)) * (len(self.tiles[0]) + 1)
        full_h = (self.big_grid_size - (self.overlap * self.scale)) * (len(self.tiles) + 1)
        self.result = PIL.Image.new('RGBA', (full_h, full_w))

    def _grid(self) -> List[List[PIL.Image.Image]]:
        return grid(self.image, self.grid_size, self.overlap)

    def __len__(self):
        return len(self.tiles) * len(self.tiles[0])

    def upscale_tile(self, idx):
        x_idx = idx % len(self.tiles[0])
        y_idx = idx // len(self.tiles[0])
        tile_to_upscale = self.tiles[y_idx][x_idx]
        upscaled_tile = tile_to_upscale.resize((self.big_grid_size, self.big_grid_size))
        tensor = ToTensor()(upscaled_tile).unsqueeze(0)

        inputs = {self.model.get_inputs()[0].name: tensor.numpy()}
        img_hr = torch.tensor(self.model.run(None, inputs)[0].squeeze(0))
        img_hr = to_image(img_hr.clip(0, 1))

        self.result.paste(img_hr,
                          (
                              y_idx * (self.big_grid_size - (self.overlap * self.scale)),
                              x_idx * (self.big_grid_size - (self.overlap * self.scale))
                          ), self.alpha_circle
                          )


def main(args):
    upscaler = Upscaler(sys.argv[-1], args.model, args.scale, args.grid_size, 2, args.device)
    for i in range(len(upscaler)):
        print(f'upscaling {i} / {len(upscaler)}')
        upscaler.upscale_tile(i)

    upscaler.result.save(args.output)


def get_arguments():
    parser = argparse.ArgumentParser(description='Upscale !')
    parser.add_argument('--scale', help='the upscale', default=4, type=int)
    parser.add_argument('--device', help='cpu or cuda', default='cuda')
    parser.add_argument('--grid-size', help='bigger = faster, but more memory used', default=64, type=int)
    parser.add_argument('--model', help='the model to use',
                        default='/home/lleonard/dev/perso/super_res/yaau/models/super_res/super_res_painting.pth')
    parser.add_argument('--output', help='output', default='./result.png')
    return parser.parse_args(sys.argv[1:-1])


if __name__ == '__main__':
    main(get_arguments())
