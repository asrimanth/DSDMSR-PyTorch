import os
import numpy as np
import cv2
from pathlib import Path
import struct
import argparse
from tqdm import tqdm

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


# https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
def get_random_crop(image, crop_height, crop_width):
    
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


def normalize(image_np):
    image_min = np.min(image_np, keepdims=True)
    image_max = np.max(image_np, keepdims=True)

    scaled_data = (image_np - image_min) / (image_max - image_min)
    return scaled_data


def create_patches_from_data(parent_dir_path, dest_path):
    counter = 0
    for folder in os.listdir(parent_dir_path):
        depths_path = parent_dir_path + "/" + folder + "/" + "depths"
        images_cam_4, images_cam_5, images_cam_6, images_cam_7 = os.listdir(depths_path)
        image_path_cam_4 = depths_path + "/" + images_cam_4
        image_path_cam_5 = depths_path + "/" + images_cam_5
        
        # Image size is not capable for 512x512 cropping
        image_path_cam_6 = depths_path + "/" + images_cam_6
        image_path_cam_7 = depths_path + "/" + images_cam_7
        print(f"Indexing {folder}")
        for image_name in tqdm(os.listdir(image_path_cam_4)):
            current_image_path = image_path_cam_4 + "/" + image_name
            image = read_pfm(current_image_path)
            if image.shape[0] > 512 and image.shape[1] > 512 and image_name.endswith(".pfm"):
                image_x16 = get_random_crop(image, 512, 512)
                image_x16 = normalize(image_x16) * 255.0
                image_x8 = cv2.resize(image_x16, (256, 256))
                image_x4 = cv2.resize(image_x16, (128, 128))
                image_x2 = cv2.resize(image_x16, (64, 64))
                image_x1 = cv2.resize(image_x16, (32, 32))
                filename = image_name.split(".")[0]
                cv2.imwrite(dest_path + "/x16/" + images_cam_4 + "_" + filename + "x16" + ".png", image_x16)
                cv2.imwrite(dest_path + "/x8/" + images_cam_4 + "_" + filename + "x8" + ".png", image_x8)
                cv2.imwrite(dest_path + "/x4/" + images_cam_4 + "_" + filename + "x4" + ".png", image_x4)
                cv2.imwrite(dest_path + "/x2/" + images_cam_4 + "_" + filename + "x2" + ".png", image_x2)
                cv2.imwrite(dest_path + "/x1/" + images_cam_4 + "_" + filename + "x1" + ".png", image_x1)
                counter += 1
            
        for image_name in tqdm(os.listdir(image_path_cam_5)):
            current_image_path = image_path_cam_5 + "/" + image_name
            image = read_pfm(current_image_path)
            if image.shape[0] > 512 and image.shape[1] > 512 and image_name.endswith(".pfm"):
                image = read_pfm(current_image_path)
                image_x16 = get_random_crop(image, 512, 512)
                image_x16 = normalize(image_x16) * 255.0
                image_x8 = cv2.resize(image_x16, (256, 256))
                image_x4 = cv2.resize(image_x16, (128, 128))
                image_x2 = cv2.resize(image_x16, (64, 64))
                image_x1 = cv2.resize(image_x16, (32, 32))
                filename = image_name.split(".")[0]
                cv2.imwrite(dest_path + "/x16/" + images_cam_5 + "_" + filename + "x16" + ".png", image_x16)
                cv2.imwrite(dest_path + "/x8/" + images_cam_5 + "_" + filename + "x8" + ".png", image_x8)
                cv2.imwrite(dest_path + "/x4/" + images_cam_5 + "_" + filename + "x4" + ".png", image_x4)
                cv2.imwrite(dest_path + "/x2/" + images_cam_5 + "_" + filename + "x2" + ".png", image_x2)
                cv2.imwrite(dest_path + "/x1/" + images_cam_5 + "_" + filename + "x1" + ".png", image_x1)
                counter += 1
                
    print(f"{counter} datapoints created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--SOURCE-DIR', type=str, required=True)
    parser.add_argument('--DEST-DIR', type=str, required=True)
    args = parser.parse_args()
    parent_dir_path = args.SOURCE_DIR # "/l/vision/jolteon_ssd/cw234/ETH/eth3d"
    dest_path = args.DEST_DIR # /l/vision/v5/sragas/ETH_3D
    create_patches_from_data(parent_dir_path, dest_path)
    