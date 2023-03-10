import pathlib
from typing import Union

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner, NNFramework


def load_image(img_filepath: Union[str, pathlib.Path], img_shape: tuple) -> np.array:
    """
    Loading image from filepath to numpy array in RGB format

    Args:
        img_filepath (str, pathlib.Path): filepath to image
        img_shape (tuple): output image shape, tuple of integers

    Returns:
        img (np.array): image as numpy matrix

    """
    img = Image.open(str(img_filepath)).resize((img_shape[0], img_shape[1])).convert('RGB')
    return np.array(img).reshape((img_shape[0], img_shape[1], 3))


def load_images(input_dirpath: Union[str, pathlib.Path], img_shape: tuple, img_extension: str = '*.jpg') -> np.array:
    """
    Loading all image from given input dirpath with proper file extension

    Args:
        input_dirpath (str, pathlib.Path): dirpath to folder with images
        img_shape (tuple): output image shape, tuple of integers
        img_extension (str): image file extension

    Returns:
        x_images (np.array): matrix of the images with shape (num_images, img_shape[0],  img_shape[1], 3)

    """
    input_dirpath = pathlib.Path(input_dirpath)
    images = input_dirpath.glob(f'**/{img_extension}')
    x_images = np.array([load_image(x, img_shape) for x in images])
    return x_images


def processing_images(x_input: np.array, frac: float = 0.25) -> np.array:
    """
    Preparing images before converting model to *.hef format

    Args:
        x_input (np.array): images
        frac (float): percent (in float number) of images selected to be used

    Returns:
        x_input_transformed (np.array): transformed images

    """
    if 0 < frac <= 1:
        idx = np.arange(x_input.shape[0]).astype(np.int16)
        np.random.seed(42)
        np.random.shuffle(idx)
        x_input_transformed = x_input[idx[:int(x_input.shape[0] * frac)]].astype(np.float32) / 255
    else:
        x_input_transformed = x_input.astype(np.float32) / 255

    return x_input_transformed


def convert_model_to_har(model_dirpath: pathlib.Path, x_images: np.array) -> tuple:
    """
    Converting tensorflow model to *.har format

    Args:
        model_dirpath (pathlib.Path): dirpath to tensorflow model
        x_images (np.array): images used to convert model

    Returns:
        runner (ClientRunner): runner object for *.har model
        har_filepath (pathlib.Path): filepath to *.har model

    """
    model_name = model_dirpath.name
    model_filepath = model_dirpath / 'saved_model.pb'
    runner = ClientRunner(hw_arch='hailo8')
    runner.translate_tf_model(model_filepath, model_name, nn_framework=NNFramework.TENSORFLOW2)

    runner.quantize(x_images, batch_size=32, calib_num_batch=32)
    har_filepath = model_dirpath / f'{runner.model_name}_quantized.har'
    runner.save_har(har_filepath)

    return runner, har_filepath


def save_hef(hef, name: str) -> pathlib.Path:
    hef_filename = '{}.hef'.format(name)

    output_dirpath = pathlib.Path('models')
    if not output_dirpath.exists():
        output_dirpath.mkdir(parents=True)
    hef_filepath = output_dirpath / hef_filename

    with open(hef_filepath, 'wb') as file_bin:
        file_bin.write(hef)

    return pathlib.Path(hef_filepath)


if __name__ == '__main__':
    pass
