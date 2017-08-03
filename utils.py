import keras
# from keras import backend as K
# from keras.utils.data_utils import get_file
# from keras.utils import np_utils
# from keras.utils.np_utils import to_categorical
# from keras.models import Sequential, Model
# from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
# from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
# from keras.layers.core import Flatten, Dense, Dropout, Lambda
# from keras.regularizers import l2, activity_l2, l1, activity_l1
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import SGD, RMSprop, Adam
# from keras.utils.layer_utils import layer_from_config
# from keras.metrics import categorical_crossentropy, categorical_accuracy
# from keras.layers.convolutional import *
from keras.preprocessing import image
# from keras.preprocessing.text import Tokenizer


def get_batches(
        dirname, 
        gen=image.ImageDataGenerator(), 
        shuffle=False, 
        save_to_dir=None,
        batch_size=32, 
        class_mode='categorical',
        target_size=(224,224)):
    return gen.flow_from_directory(
            dirname, 
            save_to_dir=save_to_dir,
            target_size=target_size,
            class_mode=class_mode, 
            shuffle=shuffle, 
            batch_size=batch_size)


