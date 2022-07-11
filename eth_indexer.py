import pandas as pd
import os
import argparse
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_train_and_valid_sets(parent_dir_path):
    x1, x16, x2, x4, x8 = sorted(os.listdir(parent_dir_path))
    dataset = {"image_x1_path": [], "image_x1_filename": [], "image_x1_dim": [],
              "image_x2_path": [], "image_x2_filename": [], "image_x2_dim": [],
              "image_x4_path": [], "image_x4_filename": [], "image_x4_dim": [],
              "image_x8_path": [], "image_x8_filename": [], "image_x8_dim": [],
              "image_x16_path": [], "image_x16_filename": [], "image_x16_dim": [],}
    x1_images = os.listdir(parent_dir_path + "/" + x1)
    x2_images = os.listdir(parent_dir_path + "/" + x2)
    x4_images = os.listdir(parent_dir_path + "/" + x4)
    x8_images = os.listdir(parent_dir_path + "/" + x8)
    x16_images = os.listdir(parent_dir_path + "/" + x16)
    n_images = len(x1_images)
    for index in tqdm(range(n_images)):
        image_x1_path = parent_dir_path + "/" + x1 + "/" + x1_images[index]
        image_x2_path = parent_dir_path + "/" + x2 + "/" + x2_images[index]
        image_x4_path = parent_dir_path + "/" + x4 + "/" + x4_images[index]
        image_x8_path = parent_dir_path + "/" + x8 + "/" + x8_images[index]
        image_x16_path = parent_dir_path + "/" + x16 + "/" + x16_images[index]
        
        image_x1 = cv2.imread(image_x1_path)
        image_x2 = cv2.imread(image_x2_path)
        image_x4 = cv2.imread(image_x4_path)
        image_x8 = cv2.imread(image_x8_path)
        image_x16 = cv2.imread(image_x16_path)
        
        dataset["image_x1_dim"].append(f"{image_x1.shape[0]}x{image_x1.shape[1]}")
        dataset["image_x2_dim"].append(f"{image_x2.shape[0]}x{image_x2.shape[1]}")
        dataset["image_x4_dim"].append(f"{image_x4.shape[0]}x{image_x4.shape[1]}")
        dataset["image_x8_dim"].append(f"{image_x8.shape[0]}x{image_x8.shape[1]}")
        dataset["image_x16_dim"].append(f"{image_x16.shape[0]}x{image_x16.shape[1]}")
        
        dataset["image_x1_path"].append(image_x1_path)
        dataset["image_x2_path"].append(image_x2_path)
        dataset["image_x4_path"].append(image_x4_path)
        dataset["image_x8_path"].append(image_x8_path)
        dataset["image_x16_path"].append(image_x16_path)
        
        dataset["image_x1_filename"].append(x1_images[index])
        dataset["image_x2_filename"].append(x2_images[index])
        dataset["image_x4_filename"].append(x4_images[index])
        dataset["image_x8_filename"].append(x8_images[index])
        dataset["image_x16_filename"].append(x16_images[index])
    
    dataset = pd.DataFrame(dataset)
    train, valid = train_test_split(dataset, test_size=0.2, random_state=42)
    train.to_csv("train.csv")
    valid.to_csv("valid.csv")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SOURCE-DIR", type=str, required=True)
    args = parser.parse_args()
    parent_dir_path = args.SOURCE_DIR # "/l/vision/v5/sragas/ETH_3D"
    create_train_and_valid_sets(parent_dir_path)