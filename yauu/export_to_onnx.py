import argparse

import torch


def export(model_path, image_size, output_file):
    fake_data = torch.randn(1, 3, image_size, image_size).cuda()
    model = torch.load(model_path)['model']
    torch.onnx.export(model, fake_data, output_file, export_params=True, verbose=True,
                      input_names=['lr_image'],
                      output_names=['hr_image'],
                      dynamic_axes={
                          'lr_image': {0: 'batch_size'},
                          'hr_image': {0: 'batch_size'},
                      })


def get_arguments():
    parser = argparse.ArgumentParser(description='Train an upscaler')
    parser.add_argument('--model', help='input')
    parser.add_argument('--size', help='image size', type=int)
    parser.add_argument('--output', help='output')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    export(args.model, args.size, args.output)
