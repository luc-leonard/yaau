import PIL.Image
import sys
import torch
from fastai.torch_core import show_image
from fastai.vision.core import PILImage, to_image
from torchvision.transforms import ToTensor
from typing import List
from PIL import ImageDraw


# TODO: use args for this
scale = 4
grid_size = 64
overlap = grid_size // 2
big_grid_size = (grid_size + overlap) * scale
device = 'cuda'


def grid(image: PIL.Image.Image, grid_size) -> List[List[PIL.Image.Image]]:
    the_grid = []
    h, w = image.shape
    for x in range(0, w - grid_size, grid_size):
        row = []
        for y in range(0, h - grid_size, grid_size):
            row.append(image.crop(
                (x - overlap // 2, y - overlap // 2, x + grid_size + overlap // 2, y + grid_size + overlap // 2)))
        the_grid.append(row)
    return the_grid


def main():
    data = torch.load('./super_res.pth', map_location=device)
    model = data['model'].to(device).eval()

    image = PILImage.create(sys.argv[1])

    the_grid = grid(image, grid_size)
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
                         ), alpha_circle)
            del tensor
            del img_hr
    result.save('result.png')


if __name__ == '__main__':
    main()
