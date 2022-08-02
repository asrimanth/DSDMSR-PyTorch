import pandas as pd
import os
import argparse
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_train_and_valid_sets(parent_dir_path):
    print(sorted(os.listdir(parent_dir_path)))
    
    image_1280x1024_folder, image_160x128_folder, image_320x256_folder, image_640x512_folder, image_80x64_folder = sorted(os.listdir(parent_dir_path))
    dataset = {"image_80x64_path": [], "image_80x64_filename": [], "image_80x64_dim": [],
               "image_160x128_path": [], "image_160x128_filename": [], "image_160x128_dim": [],
               "image_320x256_path": [], "image_320x256_filename": [], "image_320x256_dim": [],
               "image_320x256_path": [], "image_320x256_filename": [], "image_320x256_dim": [],
               "image_640x512_path": [], "image_640x512_filename": [], "image_640x512_dim": [],
               "image_1280x1024_path": [], "image_1280x1024_filename": [], "image_1280x1024_dim": [],
              }
    images_80x64 = sorted(os.listdir(parent_dir_path + "/" + image_80x64_folder))
    images_160x128 = sorted(os.listdir(parent_dir_path + "/" + image_160x128_folder))
    images_320x256 = sorted(os.listdir(parent_dir_path + "/" + image_320x256_folder))
    images_640x512 = sorted(os.listdir(parent_dir_path + "/" + image_640x512_folder))
    images_1280x1024 = sorted(os.listdir(parent_dir_path + "/" + image_1280x1024_folder))
    n_images = len(images_80x64)
    for index in tqdm(range(n_images)):
        image_64_path = parent_dir_path + "/" + image_80x64_folder + "/" + images_80x64[index]
        image_128_path = parent_dir_path + "/" + image_160x128_folder + "/" + images_160x128[index]
        image_256_path = parent_dir_path + "/" + image_320x256_folder + "/" + images_320x256[index]
        image_512_path = parent_dir_path + "/" + image_640x512_folder + "/" + images_640x512[index]
        image_1024_path = parent_dir_path + "/" + image_1280x1024_folder + "/" + images_1280x1024[index]
        
        image_x64 = cv2.imread(image_64_path)
        image_x128 = cv2.imread(image_128_path)
        image_x256 = cv2.imread(image_256_path)
        image_x512 = cv2.imread(image_512_path)
        image_x1024 = cv2.imread(image_1024_path)
        
        dataset["image_80x64_dim"].append(f"{image_x64.shape[0]}x{image_x64.shape[1]}")
        dataset["image_160x128_dim"].append(f"{image_x128.shape[0]}x{image_x128.shape[1]}")
        dataset["image_320x256_dim"].append(f"{image_x256.shape[0]}x{image_x256.shape[1]}")
        dataset["image_640x512_dim"].append(f"{image_x512.shape[0]}x{image_x512.shape[1]}")
        dataset["image_1280x1024_dim"].append(f"{image_x1024.shape[0]}x{image_x1024.shape[1]}")
        
        dataset["image_80x64_path"].append(image_64_path)
        dataset["image_160x128_path"].append(image_128_path)
        dataset["image_320x256_path"].append(image_256_path)
        dataset["image_640x512_path"].append(image_512_path)
        dataset["image_1280x1024_path"].append(image_1024_path)
        
        dataset["image_80x64_filename"].append(images_80x64[index])
        dataset["image_160x128_filename"].append(images_160x128[index])
        dataset["image_320x256_filename"].append(images_320x256[index])
        dataset["image_640x512_filename"].append(images_640x512[index])
        dataset["image_1280x1024_filename"].append(images_1280x1024[index])
    
    dataset = pd.DataFrame(dataset)
    train, valid = train_test_split(dataset, test_size=0.2, random_state=42)
    train.to_csv("train_dtu.csv")
    valid.to_csv("valid_dtu.csv")
    train[:3000].to_csv("train_dtu_sub.csv")
    valid[:300].to_csv("valid_dtu_sub.csv")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SOURCE-DIR", type=str, required=True)
    args = parser.parse_args()
    parent_dir_path = args.SOURCE_DIR # "/l/vision/v5/sragas/DTU_patches"
    create_train_and_valid_sets(parent_dir_path)