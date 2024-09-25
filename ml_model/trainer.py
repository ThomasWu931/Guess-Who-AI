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

class Trainer:
    def __init__(self, image_folder, feature, model_dir, img_h=218, img_w=178, model_name="VGG16") -> None:
        self.image_folder = image_folder
        self.feature = feature
        self.lr = 0.0001
        self.batch_size = 100
        self.model_dir = model_dir
        self.img_h = img_h
        self.img_w = img_w
        self.model = self.initialize_model(model_name)

    def initialize_model(self, model_name):
        if model_name == "VGG16":
            base_model = keras.applications.VGG16(weights=None, input_shape=(self.img_h, self.img_w, 3), include_top=False)
        elif model_name == "ResNet50V2":
            base_model = keras.applications.ResNet50V2(weights=None, input_shape=(self.img_h, self.img_w, 3), include_top=False)
        else:
            raise Exception(f"[initialize_model] ERROR: Invalid model type {model}")
        add_model = Sequential()
        add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        add_model.add(Dense(1, activation='sigmoid'))
        model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.lr),
                        metrics=['accuracy'])
        model.summary()
        return model

    def train_and_save_model(self):
        print(f"[train_and_save_weights]: Training for feature {self.feature}")

        # Load the train and validation data
        train_datagen = ImageDataGenerator() # NOTE!?: Might want to include rescale=1.0/255
        val_datagen = ImageDataGenerator()

        # Flow images from the directory
        train_generator = train_datagen.flow_from_directory(
            f'{self.image_folder}/Train',
            target_size=(self.img_h, self.img_w),   
            batch_size=self.batch_size,
            class_mode='binary'       
        )
        val_generator = val_datagen.flow_from_directory(
            f'{self.image_folder}/Valid',
            target_size=(self.img_h, self.img_w),   
            batch_size=self.batch_size,
            class_mode='binary'       
        )

        # Train and save
        model_path = f"{self.model_dir}/{self.feature}.keras"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            filepath=model_path,  # Filepath where the model will be saved
            monitor='val_accuracy',  # Monitor the validation accuracy
            save_best_only=False,    # Save the model at every epoch
            save_weights_only=False, # Save the entire model, not just the weights
            mode='max',              # Mode to monitor the accuracy
            verbose=1                # Print a message when saving the model
        )

        # Fit the model
        self.model.fit(
            x=train_generator, 
            epochs=2,
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
t = Trainer(image_folder="./ml_model/data/Blond_Hair", 
            feature="Blond_Hair",
            model_dir="./ml_model/models"    
        )
t.train_and_save_model()


