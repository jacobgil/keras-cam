import cv2
import glob
import os
import numpy as np
from keras.utils.np_utils import to_categorical

def load_inria_person(path):
    pos_path = os.path.join(path, "pos")
    neg_path = os.path.join(path, "/neg")
    pos_images = [cv2.resize(cv2.imread(x), (64, 128)) for x in glob.glob(pos_path + "/*.png")]
    pos_images = [np.transpose(img, (2, 0, 1)) for img in pos_images]
    neg_images = [cv2.resize(cv2.imread(x), (64, 128)) for x in glob.glob(neg_path + "/*.png")]
    neg_images = [np.transpose(img, (2, 0, 1)) for img in neg_images]
    y = [1] * len(pos_images) + [0] * len(neg_images)
    y = to_categorical(y, 2)
    X = np.float32(pos_images + neg_images)
    
    return X, y
