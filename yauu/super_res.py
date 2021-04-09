import sys
from typing import List

import PIL.Image
import torch
from fastai.torch_core import show_image
from fastai.vision.core import PILImage, to_image
from torchvision.transforms import ToTensor


def grid(image: PIL.Image.Image, grid_size) -> List[PIL.Image.Image]:
    the_grid = []
    h, w = image.shape
    for x in range(0, w - grid_size, grid_size):
        row = []
        for y in range(0, h - grid_size, grid_size):
            row.append(image.crop((x, y, x + grid_size, y + grid_size)))
        the_grid.append(row)
    return the_grid

scale = 4
grid_size = 64
big_grid_size = grid_size * scale
device = 'cuda'

def main():
    data = torch.load('./super_res_mse.pth', map_location=device)
    model = data['model'].to(device).eval()

    image = PILImage.create(sys.argv[1])

    the_grid = grid(image, grid_size)
    full_w = big_grid_size * len(the_grid[0])
    full_h = big_grid_size * len(the_grid)
    result = PIL.Image.new('RGB', (full_h, full_w))
    for x, row in enumerate(the_grid):
        for y, element in enumerate(row):
            tensor = ToTensor()(element.resize((big_grid_size, big_grid_size))).unsqueeze(0).float().to(device)
            img_hr, *_ = model(tensor)
            result.paste(to_image(img_hr.clip(0, 1)), (x * img_hr.shape[1], y * img_hr.shape[2]))
            del tensor
            del img_hr
    result.save('result.png')

if __name__ == '__main__':
    main()