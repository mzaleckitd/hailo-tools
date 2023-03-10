import pathlib
import shutil

import PIL.Image
import numpy as np
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceDataType
from hailo_sdk_common.targets.inference_targets import SdkNative

from tools import load_images, processing_images


def build_runner(model_filepath: pathlib.Path):
    runner = ClientRunner(har_path=str(model_filepath))
    return runner


def main(data_dirpath, model_path, output_dirpath):
    if not output_dirpath.exists():
        output_dirpath.mkdir(parents=True)

    input_size = 224
    frac = 0.1

    x_images = load_images(data_dirpath, (input_size, input_size))
    x_images = np.stack(x_images)
    x_images = processing_images(x_images, frac=frac)

    runner = build_runner(model_path)
    outputs = runner.infer(target=SdkNative(), dataset=x_images, data_type=InferenceDataType.np_array, batch_size=1)

    if not output_dirpath.exists():
        output_dirpath.mkdir(parents=True)

    outputs = [PIL.Image.fromarray(np.uint8(x[:, :, 0] * 255), 'L') for x in outputs]
    for ii, output_img in enumerate(outputs):
        filepath = output_dirpath / f'image-{str(ii).zfill(4)}.png'
        output_img.save(filepath)

    shutil.make_archive('output', 'zip', output_dirpath)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', '-d', type=str, help='dirpath to data with images')
    parser.add_argument('--model_filepath', '-m', type=str, help='filepath to saved model in *.har format')
    parser.add_argument('--output_dirpath', '-o', type=str, help='dirpath to output folder')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    model_file = pathlib.Path(args.model_filepath)
    output_dir = pathlib.Path(args.output_dirpath)

    main(data_dir, model_file, output_dir)
