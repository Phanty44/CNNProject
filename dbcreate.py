import os
import cv2
from tqdm import tqdm


def create_training_data(X_P, Y_P, DATA, CATEGORIES, training_data):
    for category in CATEGORIES:  # loop through all categories of photos

        path = os.path.join(DATA, category)  # create path to subfolder in database
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image in subfolder
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array, grayscale
                new_array = cv2.resize(img_array, (X_P, Y_P))  # resize to X_P Y_P
                training_data.append([new_array, class_num])  # add transformed picture to list
            except Exception as e:
                pass
    return training_data


def create_testing_data(X_P, Y_P, TEST, testing_data):
    for img in tqdm(os.listdir(TEST)):  # iterate over each image
        try:
            img_array2 = cv2.imread(os.path.join(TEST, img), cv2.IMREAD_GRAYSCALE)  # convert to array, grayscale
            new_array2 = cv2.resize(img_array2, (X_P, Y_P))  # resize to X_P Y_P
            testing_data.append([new_array2])  # add transformed picture to list
        except Exception as e:
            pass
    return testing_data
