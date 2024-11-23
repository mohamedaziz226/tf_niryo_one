#!/usr/bin/env python

import numpy as np
import cv2
import os
import random
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, GlobalAveragePooling2D
import time

# Assurez-vous d'avoir un fichier "utils.py" avec la fonction "standardize_img" si nécessaire
import utils  # Assuming you have a utils.py file with necessary functions

class MyModel:
    def __init__(self, input_shape, output_len):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        self.model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),  # Use GlobalAveragePooling2D
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),  # Add another dropout layer
            Dense(output_len, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",  # Use categorical_crossentropy
            metrics=["accuracy"]
        )


def load_dataset(data_path):
    objects_names = os.listdir(data_path)  # Liste des noms d'objets
    print(f"Object names: {objects_names}")  # Debugging: Show the object names

    objects_list = []
    labels_list = []
    files_names = []
    obj_id = 0

    try:
        os.mkdir("./data_mask")
    except FileExistsError:
        pass

    for name in objects_names:
        list_dir = os.listdir(os.path.join(data_path, name))
        print(f"{name} contains {len(list_dir)} files.")  # Debugging: Number of files in each object folder

        try:
            os.mkdir(f"./data_mask/{name}")
        except FileExistsError:
            pass

        for file_name in list_dir:
            img_path = os.path.join(data_path, name, file_name)
            img = cv2.imread(img_path)

            if img is None:  # Vérifiez si l'image est valide
                print(f"Warning: Image {img_path} could not be read.")
                continue

            # Convertir en HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Définir les plages HSV pour le masque
            lower_hole = np.array([0, 0, 0])
            upper_hole = np.array([179, 255, 50])
            mask = cv2.inRange(img_hsv, lower_hole, upper_hole)

            # Vérification de l'existence des contours
            _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #if len(contours) == 2:
                #contours, _ , _ = contours
            #print(contours)


            if not contours:  # Si aucun contour n'est trouvé
                #print(f"No contours found for image {file_name}. Skipping.")
                continue

    for contour in contours:
    # Vérifier que le contour a des points avant de calculer l'aire
     if contour is not None and len(contour) > 5:
        try:
            area = cv2.contourArea(contour)  # Calculer l'aire du contour
            if area > 5:  # Filtrer les petites zones
                x, y, w, h = cv2.boundingRect(contour)
                hole_img = img[y:y + h, x:x + w]
                hole_img = cv2.resize(hole_img, (96, 96))  # Redimensionner
                img_float = hole_img.astype(np.float32) / 255.0  # Normaliser

                label = np.zeros(len(objects_names), np.float32)
                label[obj_id] = 1

                objects_list.append(img_float)
                labels_list.append(label)
                files_names.append(file_name)
        except cv2.error as e:
            print(f"Erreur de contour pour l'image {file_name}: {e}")
            continue  # Continuer avec le suivant en cas d'erreur



        #obj_id += 1

    print(f"Loaded {len(objects_list)} images.")  # Debugging: Number of images loaded
    return objects_list, labels_list, files_names, objects_names


def shuffle(*list_to_shuffle):
    # shuffle data
    c = list(zip(*list_to_shuffle))
    random.shuffle(c)
    return zip(*c)


def test(model, objects_list, labels_list, objects_names, training_size, files_names):
    print("\n\ntesting...", end="", flush=True)
    t = time.time()
    predictions = model.model.predict(objects_list)
    t = time.time() - t
    print("ok ", str(t)[2:5] + "ms for " + str(len(objects_list)) + " images")

    nb_error = 0
    nb_error_new = 0
    for x in range(len(objects_list)):
        x_max, y_max = predictions[x].argmax(), labels_list[x].argmax()
        if x == training_size:
            print("training data end")
        if x_max != y_max:
            if x > training_size:
                nb_error_new += 1
            else:
                nb_error += 1
            print("error", x, predictions[x], y_max, x_max, objects_names[y_max],
                  objects_names[x_max], files_names[x])

    acc_tot = (len(objects_list) - nb_error) / len(objects_list)
    acc_test = (len(objects_list) - training_size - nb_error_new) / max(
        len(objects_list) - training_size, 1)
    print(acc_tot * 100, "%", "training data (sample size " + str(training_size) + ")")
    print(acc_test * 100, "%", "new data (sample size " + str(
        len(objects_list) - training_size) + ")")
    return acc_tot, acc_test


def training():
    np.set_printoptions(precision=3, suppress=True)

    objects_list, labels_list, files_names, objects_names = load_dataset("data")
    objects_list, labels_list, files_names = shuffle(objects_list, labels_list, files_names)

    if len(objects_list) == 0:
        print("cannot train without a data set of 0")
        return None

    objects_list = np.array(objects_list)
    labels_list = np.array(labels_list)

    training_size = int(len(objects_list) * 0.8)

    object_train = objects_list[:training_size]
    labels_train = labels_list[:training_size]

    


    print("object train",object_train)
    print("label train",labels_train)



    print(f"Total images: {len(objects_list)}")
    print("Object train shape:", object_train.shape)
    print("Label train shape:", labels_train.shape)


    model = MyModel((96, 96, 3), len(objects_names))  # MobileNetV2 input shape
    model.model.summary()  # Print model summary

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=5,
        width_shift_range=[-4, 4],
        height_shift_range=[-4, 4],
        zoom_range=[0.95, 1.05],
        horizontal_flip=0.5,
        vertical_flip=0.5,
        fill_mode="nearest"  # Use 'nearest' fill mode
    )

    history = model.model.fit(
        datagen.flow(object_train, labels_train, batch_size=10),
        steps_per_epoch=len(object_train) // 10,  # Correct steps_per_epoch
        epochs=25
    )

    print("Saving model...", end="", flush=True)
    model.model.save("model")
    print("Model saved.")

    test(model, objects_list, labels_list, objects_names, training_size, files_names)
    return model.model


if __name__ == '__main__':
    training()
