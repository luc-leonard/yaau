import argparse
import sys
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


def main(args):
    overlap = args.grid_size // 2
    big_grid_size = (args.grid_size + overlap) * args.scale

    data = torch.load(args.model, map_location=args.device)
    model = data['model'].to(args.device).eval()

    image = PILImage.create(sys.argv[-1])

    the_grid = grid(image, args.grid_size, overlap)
    full_w = (big_grid_size - (overlap * args.scale)) * len(the_grid[0])
    full_h = (big_grid_size - (overlap * args.scale)) * len(the_grid)
    result = PIL.Image.new('RGBA', (full_h, full_w))

    alpha_circle = PIL.Image.new('1', (big_grid_size, big_grid_size))
    ImageDraw.Draw(alpha_circle).ellipse([(0, 0), ((big_grid_size, big_grid_size))], fill=(255,))

    for x, row in enumerate(the_grid):
        for y, element in enumerate(row):
            tensor = ToTensor()(element.resize((big_grid_size, big_grid_size))).unsqueeze(0).float().to(args.device)
            img_hr, *_ = model(tensor)
            img_hr = to_image(img_hr.clip(0, 1))
            result.paste(img_hr,
                         (
                             x * (big_grid_size - (overlap * args.scale)),
                             y * (big_grid_size - (overlap * args.scale))
                         ), alpha_circle)
            del tensor
            del img_hr
    result.save(args.output)


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
