import pathlib
from typing import Union

import numpy as np

from tools import convert_model_to_har, load_images, processing_images, save_hef


def convert_arc_model(model_dirpath_arc: Union[str, pathlib.Path], data_dirpath: Union[str, pathlib.Path],
                      input_size: int, frac: float) -> pathlib.Path:

    model_dirpath_arc = pathlib.Path(model_dirpath_arc)
    data_dirpath = pathlib.Path(data_dirpath)

    x_images = load_images(data_dirpath, (input_size, input_size))
    x_images = np.stack(x_images)
    x_images = processing_images(x_images, frac=frac)   # take a sample of all images

    runner_arc, har_filepath_arc = convert_model_to_har(model_dirpath_arc, x_images)
    print(f'Save output model into: {har_filepath_arc}')
    hef = runner_arc.get_hw_representation()
    hef_filepath = save_hef(hef, runner_arc.model_name)

    print(f'Save output model into: {hef_filepath}')
    return hef_filepath


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--arc_model_dir', '-m', type=str, help='dirpath to saved arc model')
    parser.add_argument('--frac', '-f', type=float, default=0.1,
                        help='fractional value of the number of images from data_dir')
    parser.add_argument('--data_dir', '-d', type=str, help='dirpath to data with images')
    parser.add_argument('--input_size', '-i', type=int, default=224,
                        help='image input size (s, s, 3) for tensorflow model')

    args = parser.parse_args()

    print('MODEL ARC DIR:', args.arc_model_dir)
    print('DATA DIR:', args.data_dir)
    print('SIZE:', args.input_size)
    print('FRAC:', args.frac)

    convert_arc_model(
        args.arc_model_dir, args.data_dir, args.input_size, args.frac
    )
