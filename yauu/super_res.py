import argparse
import sys
from pathlib import Path
from typing import List

import PIL.Image
import torch
from PIL import ImageDraw
from fastai.vision.core import PILImage, to_image
from torchvision.transforms import ToTensor


def grid(image: PIL.Image.Image, grid_size, overlap) -> List[List[PIL.Image.Image]]:
    the_grid = []
    h, w = image.shape
    for x in range(0, w - grid_size, grid_size):
        row = []
        for y in range(0, h - grid_size, grid_size):
            row.append(image.crop(
                (x - overlap // 2, y - overlap // 2, x + grid_size + overlap // 2, y + grid_size + overlap // 2)))
        the_grid.append(row)
    return the_grid


def super_res(file_path: Path, scale: int, grid_size: int, overlap_factor, model: Path, device: str) -> PIL.Image:
    overlap = grid_size // overlap_factor

    big_grid_size = (grid_size + overlap) * scale

    data = torch.load(model, map_location=device)
    model = data['model'].to(device).eval()

    image = PILImage.create(file_path)

    the_grid = grid(image, grid_size, overlap)
    full_w = (big_grid_size - (overlap * scale)) * len(the_grid[0])
    full_h = (big_grid_size - (overlap * scale)) * len(the_grid)

    result = PIL.Image.new('RGBA', (full_h, full_w))

    alpha_circle = PIL.Image.new('1', (big_grid_size, big_grid_size))
    ImageDraw.Draw(alpha_circle).ellipse([(0, 0), ((big_grid_size, big_grid_size))], fill=(255,))

    for x, row in enumerate(the_grid):
        for y, element in enumerate(row):
            tensor = ToTensor()(element.resize((big_grid_size, big_grid_size))).unsqueeze(0).float().to(device)
            img_hr, *_ = model(tensor)
            img_hr = to_image(img_hr.clip(0, 1))
            result.paste(img_hr,
                         (
                             x * (big_grid_size - (overlap * scale)),
                             y * (big_grid_size - (overlap * scale))
                         ), alpha_circle
                         )
            del tensor
            del img_hr
    return result


class Upscaler:
    def __init__(self, image_path: Path, model, scale: int, grid_size: int, overlap_factor: int, device: str):
        # the image to upscale
        self.image = PILImage.create(image_path)
        # the model to use
        data = torch.load(model, map_location=device)
        self.model = data['model'].to(device).eval()
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

        full_w = (self.big_grid_size - (self.overlap * self.scale)) * len(self.tiles[0])
        full_h = (self.big_grid_size - (self.overlap * self.scale)) * len(self.tiles)
        self.result = PIL.Image.new('RGBA', (full_h, full_w))

    def _grid(self) -> List[List[PIL.Image.Image]]:
        return grid(self.image, self.grid_size, self.overlap)

    def __len__(self):
        return len(self.tiles) * len(self.tiles[0])

    def upscale_tile(self, idx):
        x_idx = idx % len(self.tiles[0])
        y_idx = idx // len(self.tiles[0])
        tile_to_upscale = self.tiles[y_idx][x_idx]
        tile_to_upscale.resize((self.big_grid_size, self.big_grid_size))
        tensor = ToTensor()(tile_to_upscale.resize((self.big_grid_size, self.big_grid_size))).unsqueeze(0).float().to(
            self.device)
        img_hr, *_ = self.model(tensor)
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
