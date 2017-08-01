import cv2

import glob
import os
import numpy as np
from keras.utils.np_utils import to_categorical
from preprocessing import preprocess_image_batch

CLASS_NAME_MAPPING = {}
PER_CLASS_MAX_IMAGES = 24


def load_data(path): 
    data_set_input_images_files, data_set_input_images_true_label = get_class_wise_images_and_true_label(path)
    processed_input_images = [preprocess_image_batch([image])
                               for image in data_set_input_images_files]

    global CLASS_NAME_MAPPING
    nb_classes = len(CLASS_NAME_MAPPING.keys())
    X_train = np.concatenate(processed_input_images)

    y_out = np.concatenate(data_set_input_images_true_label)
    y_out = to_categorical(y_out, nb_classes=nb_classes)  # to get sofmax shape of (None, nb_classes)
    Y_train = y_out


    from sklearn.utils import shuffle
    X_train, Y_train = shuffle(X_train, Y_train)

    return X_train, Y_train, nb_classes


def get_class_wise_images_and_true_label(path):
    print('path', path+"/*")
    directory = glob.glob(path + '/*')
    data_set_input_images = []
    data_set_input_images_true_label = []
    global CLASS_NAME_MAPPING
    index = 0
    for sub_directory in directory:
        if os.path.isdir(sub_directory):
            class_dir_name = sub_directory.split('/')[-1]
            CLASS_NAME_MAPPING[index] = class_dir_name
            image_class_files = glob.glob(sub_directory + '/*.jpeg')[:PER_CLASS_MAX_IMAGES]
            data_set_input_images.extend(image_class_files)
            data_set_input_images_true_label.extend([[index]] * len(image_class_files))
            index += 1
    return data_set_input_images, data_set_input_images_true_label

def load_inria_person(path):
    pos_path = os.path.join(path, "pos")
    neg_path = os.path.join(path, "neg")
    pos_images = [cv2.resize(cv2.imread(x), (64, 128)) for x in glob.glob(pos_path + "/*.jpeg")]
    pos_images = [np.transpose(img, (2, 0, 1)) for img in pos_images]
    neg_images = [cv2.resize(cv2.imread(x), (64, 128)) for x in glob.glob(neg_path + "/*.jpeg")]
    neg_images = [np.transpose(img, (2, 0, 1)) for img in neg_images]
    y = [1] * len(pos_images) + [0] * len(neg_images)
    y = to_categorical(y, 2)
    X = np.float32(pos_images + neg_images)
    
    return X, y
