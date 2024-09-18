from tensorflow.keras.preprocessing import image
import numpy as np
import keras
from tensorflow.keras import layers, models
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from random import seed 
from random import randint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd
from keras.optimizers.legacy import Adam # Using legacy since modern one slower on Macs
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import shutil

class FeatureDataPartitioner:
    def __init__(self, image_folder, label_csv_dir, feature, output_data_folder) -> None:
        self.image_folder = image_folder
        self.label_csv_dir = label_csv_dir
        self.feature = feature
        self.label_df = pd.read_csv(label_csv_dir)
        self.label_df.set_index('image_id', inplace=True)
        self.output_data_folder = output_data_folder

    def get_even_feature_split(self):
        """Look through all images and get the maximum subset which includes a 50/50 split between data with and without the feature"""
        with_feature  = self.label_df[self.label_df[self.feature] == 1]
        without_feature = self.label_df[self.label_df[self.feature] == -1]     
   
        split_size = min(len(without_feature), len(with_feature))
        print(f"[get_even_feature_split] Split size is {split_size}")
        with_feature_sample = with_feature.sample(n=split_size)
        without_feature_sample = without_feature.sample(n=split_size)

        with_feature_train = with_feature_sample.sample(frac=0.9)
        with_feature_val = with_feature_sample.drop(with_feature_train.index)

        without_feature_train = without_feature_sample.sample(frac=0.9)
        without_feature_val = without_feature_sample.drop(without_feature_train.index)

        return [df.index for df in [with_feature_train, with_feature_val, without_feature_train, without_feature_val]]

    def upload_imgs(self, image_file_names: list[str], input_folder_dir, output_folder_dir):
        for file_name in image_file_names:
            os.makedirs(output_folder_dir, exist_ok=True)
            shutil.copyfile(f"{input_folder_dir}/{file_name}", f"{output_folder_dir}/{file_name}")    

    def process_and_upload_data(self):
        with_feature_train, with_feature_val, without_feature_train, without_feature_val = self.get_even_feature_split()
        print("[process_and_upload_data] Uploading")
        self.upload_imgs(with_feature_train, self.image_folder, f"{self.output_data_folder}/Train/yes_{self.feature}")
        self.upload_imgs(with_feature_val, self.image_folder, f"{self.output_data_folder}/Valid/yes_{self.feature}")
        self.upload_imgs(without_feature_train, self.image_folder, f"{self.output_data_folder}/Train/no_{self.feature}")
        self.upload_imgs(without_feature_val, self.image_folder, f"{self.output_data_folder}/Valid/no_{self.feature}")

f = FeatureDataPartitioner("./raw_data/img_align_celeba/img_align_celeba", "./raw_data/list_attr_celeba.csv", "Brown_Hair", "data/Brown_Hair")
f.process_and_upload_data()
