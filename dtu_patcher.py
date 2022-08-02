import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

from pathlib import Path
import struct


# https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = max_x if max_x in (0, 1) else np.random.randint(0, max_x)
    y = max_y if max_y in (0, 1) else np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop


def normalize(image_np):
    image_min = np.min(image_np, keepdims=True)
    image_max = np.max(image_np, keepdims=True)

    scaled_data = (image_np - image_min) / (image_max - image_min + 1e-8)
    return scaled_data


# https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale


def create_patches_from_data(parent_dir_path, dest_path):
    for folder in tqdm(os.listdir(parent_dir_path)):
        current_path = parent_dir_path + "/" + folder
        if folder[-1].isdigit():
            for image_name in os.listdir(current_path):
                if image_name.endswith(".pfm"):
                    image_path = current_path + "/" + image_name
                    image_hr = read_pfm(image_path)
                    image_hr = normalize(image_hr) * 255.0
                    if image_hr.shape != (1200, 1600):
                        print(image_hr.shape)
                    image_1280x1024 = cv2.resize(image_hr, (1280, 1024), interpolation=cv2.INTER_LINEAR)
                    image_640x512 = cv2.resize(image_1280x1024, (640, 512), interpolation=cv2.INTER_LINEAR)
                    image_320x256 = cv2.resize(image_1280x1024, (320, 256), interpolation=cv2.INTER_LINEAR)
                    image_160x128 = cv2.resize(image_1280x1024, (160, 128), interpolation=cv2.INTER_LINEAR)
                    image_80x64 = cv2.resize(image_1280x1024, (80, 64), interpolation=cv2.INTER_LINEAR)
                    filename = image_name.split(".")[0]
                    cv2.imwrite(dest_path + "/img1280x1024/" + folder + "_" + filename + "_x1024" + ".png", image_1280x1024)
                    cv2.imwrite(dest_path + "/img640x512/" + folder + "_" + filename + "_x512" + ".png", image_640x512)
                    cv2.imwrite(dest_path + "/img320x256/" + folder + "_" + filename + "_x256" + ".png", image_320x256)
                    cv2.imwrite(dest_path + "/img160x128/" + folder + "_" + filename + "_x128" + ".png", image_160x128)
                    cv2.imwrite(dest_path + "/img80x64/" + folder + "_" + filename + "_x64" + ".png", image_80x64)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--SOURCE-DIR', type=str, required=True)
    parser.add_argument('--DEST-DIR', type=str, required=True)
    args = parser.parse_args()
    parent_dir_path = args.SOURCE_DIR # "/nfs/jolteon/data/ssd/vkvats/datasets/DTU/Depths_raw"
    dest_path = args.DEST_DIR # "/l/vision/v5/sragas/DTU_patches"
    create_patches_from_data(parent_dir_path, dest_path)