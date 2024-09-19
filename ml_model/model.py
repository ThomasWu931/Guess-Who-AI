import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


class MLModel:
    def __init__(self, img_h, img_w, model_path, load_weights=False):
        self.img_h = img_h
        self.img_w = img_w
        self.weights_dir = weights_dir
        self.feature = feature
        self.model = self.initialize_model()
        if load_weights:
            self.load_weights()

    def get_weights_path(self):
        return f"{self.weights_dir}/feature-{self.feature}.keras"

    def load_weights(self) -> None:
        weights_path = self.get_weights_path()
        self.model.load_weights(weights_path)

    def initialize_model(self, model="VGG16"):
        if model == "VGG16":
            base_model = keras.applications.VGG16(weights=None, input_shape=(self.img_h, self.img_w, 3), include_top=False)
        elif model == "ResNet50V2":
            base_model = keras.applications.ResNet50V2(weights=None, input_shape=(self.img_h, self.img_w, 3), include_top=False)
        else:
            raise Exception(f"[initialize_model] ERROR: Invalid model type {model}")
        add_model = Sequential()
        add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        add_model.add(Dense(1, activation='sigmoid'))
        model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001),
                        metrics=['accuracy'])
        model.summary()
        return model

    def predict(self, image_paths: list[str], show_visual=False) -> np.ndarray:
        images = []
        for img_path in image_paths:
            if img_path.endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image file
                img = image.load_img(img_path, target_size=(self.img_h, self.img_w))  # Load image
                img_array = image.img_to_array(img)  # Convert image to array
                images.append(img_array)  # Append to list
            else:
                raise Exception(f"[predict] ERROR: The following is not valid iamge format {img_path}")

        prd = self.model.predict(np.array(images))
        if show_visual:
            for path, img, p in zip(image_paths, images, prd):
                plt.figure()
                plt.imshow(img.astype('uint8'))
                plt.title(f'Image {path} with pred {p}')
                plt.axis('off')
                plt.show()

        return prd
