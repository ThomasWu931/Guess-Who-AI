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
from model import *

class Trainer:
    def __init__(self, image_folder, feature, model: MLModel) -> None:
        self.image_folder = image_folder
        self.feature = feature
        self.lr = 0.0001
        self.batch_size = 100
        self.model = model

    def train_and_save_weights(self):
        print(f"[train_and_save_weights]: Training for feature {self.model.feature}")

        # Load the train and validation data
        train_datagen = ImageDataGenerator() # NOTE!?: Might want to include rescale=1.0/255
        val_datagen = ImageDataGenerator()

        # Flow images from the directory
        train_generator = train_datagen.flow_from_directory(
            f'{self.image_folder}/{self.model.feature}/Train',
            target_size=(self.model.img_h, self.model.img_w),   
            batch_size=self.batch_size,
            class_mode='binary'       
        )
        val_generator = val_datagen.flow_from_directory(
            f'{self.image_folder}/{self.model.feature}/Valid',
            target_size=(self.model.img_h, self.model.img_w),   
            batch_size=self.batch_size,
            class_mode='binary'       
        )

        # Train and save
        filepath = self.model.get_weights_path()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = ModelCheckpoint(
            filepath=filepath,  # Filepath where the best model weights will be saved
            monitor='val_accuracy',    # Monitor the validation accuracy
            save_best_only=True,       # Save only the model with the best validation accuracy
            mode='max',                # Save when the monitored metric (accuracy) is at its maximum
            verbose=1                  # Print a message when saving the model
        )
        breakpoint()
        self.model.fit(
            x=train_generator, 
            epochs=3,
            validation_data=val_generator,
            callbacks=[checkpoint],
            steps_per_epoch=80,
            validation_steps=5,
        )

        # Display stats
        self.show_incorrect_labels(self.model, val_generator)

    def show_incorrect_labels(self, model, val_generator):
        # Reset the generator to ensure it starts from the beginning
        val_generator.reset()

        incorrect_imgs = []
        incorrect_labels = []
        incorrect_preds = []

        # while True:
        for i in range(5):
            try:
                # Get the next batch of validation data
                val_imgs, val_labels = next(val_generator)
                
                # Make predictions
                val_predictions = model.predict(val_imgs)

                # Flatten predictions and labels if necessary
                val_predictions = np.squeeze(val_predictions)
                val_labels = np.squeeze(val_labels)

                # Convert predictions to binary values (assuming binary classification)
                val_predictions_binary = (val_predictions > 0.5).astype(int)

                # Find incorrect predictions
                batch_incorrect_indices = np.where(val_predictions_binary != val_labels)[0]

                # Collect incorrect images, labels, and predictions
                incorrect_imgs.extend(val_imgs[batch_incorrect_indices])
                incorrect_labels.extend(val_labels[batch_incorrect_indices])
                incorrect_preds.extend(val_predictions_binary[batch_incorrect_indices])

            except StopIteration:
                # Generator is exhausted
                break

        # Display the incorrect labels and images
        for i in range(min(len(incorrect_imgs), 10)):  # Show a max of 5 images
            plt.figure(figsize=(4, 4))
            plt.imshow(incorrect_imgs[i].astype('uint8'))  # Ensure image is in the right format
            plt.title(f"True Label: {incorrect_labels[i]}, Predicted: {incorrect_preds[i]}")
            plt.axis('off')
            plt.show()



import tensorflow as tf

# Set parallelism threads
tf.config.threading.set_intra_op_parallelism_threads(4)  # Controls parallel threads within individual operations
tf.config.threading.set_inter_op_parallelism_threads(4)  # Controls parallel threads between operations

# ,Bald,Bangs,Black_Hair,Blond_Hair,Brown_Hair,Chubby,Gray_Hair,Male,Mouth_Slightly_Open,Pale_Skin,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,Wearing_Necklace,Wearing_Necktie
t = Trainer(image_folder="data/Brown_Hair", 
            feature="Brown_Hair",
            weights_dir="weights"    
        )
t.train_and_save_weights()


