import pathlib
from typing import Union

import numpy as np

from tools import load_images, processing_images, convert_model_to_har, save_hef


def convert_edge_line_model(model_edge_dirpath: Union[str, pathlib.Path], model_line_dirpath: Union[str, pathlib.Path],
                            data_dirpath: Union[str, pathlib.Path], input_size: tuple, frac: float) -> pathlib.Path:
    """
    Converts TensorFlow edge+line model into HAILO format (edge & line models will be calculated parallel)

    Args:
        model_edge_dirpath (str or pathlib.Path): path to dir with TF edge model (model save in "folder" format)
        model_line_dirpath (str or pathlib.Path): path to dir with TF edge model (model save in "folder" format)
        data_dirpath (str or pathlib.Path): path to dir with data for edge / line
        input_size (tuple): model input sie
        frac (float): fraction of data used to convert model, value should be in range (0, 1]

    Returns:
        hef_filepath (pathlib.Path): filepath to saved *.hef file

    """
    model_edge_dirpath = pathlib.Path(model_edge_dirpath)
    model_line_dirpath = pathlib.Path(model_line_dirpath)

    x_images = load_images(data_dirpath, input_size)
    x_images = np.stack(x_images)
    x_images = processing_images(x_images, frac=frac)   # take a sample of all images

    runner_edge, har_filepath_edge = convert_model_to_har(model_edge_dirpath, x_images)
    runner_line, har_filepath_line = convert_model_to_har(model_line_dirpath, x_images)
   
    print('har file edge:', har_filepath_edge)
    print('har file line:', har_filepath_line)

    runner_edge.join(runner_line)
    hef = runner_edge.get_hw_representation()
    hef_filepath = save_hef(hef, runner_edge.model_name)

    print(f'Save output model into: {hef_filepath}')
    return hef_filepath


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--edge_model_dir', '-e', type=str, help='dirpath to saved edge model')
    parser.add_argument('--line_model_dir', '-l', type=str, help='dirpath to saved line model')
    parser.add_argument('--data_dir', '-d', type=str, help='dirpath to data with images')
    parser.add_argument('--input_size', '-i', type=int, default=224,
                        help='image input size (s, s, 3) for tensorflow model')
    parser.add_argument('--frac', '-f', type=float, default=0.1,
                        help='fractional value of the number of images from data_dir')
    args = parser.parse_args()

    print('MODEL EDGE DIR:', args.edge_model_dir)
    print('MODEL LINE DIR:', args.line_model_dir)
    print('DATA DIR:', args.data_dir)
    print('SIZE:', args.input_size)
    print('FRAC:', args.frac)

    convert_edge_line_model(
        args.edge_model_dir, args.line_model_dir, args.data_dir, (args.input_size, args.input_size) , args.frac
    )
