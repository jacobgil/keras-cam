from keras.models import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from model import *
from data import *
from utils import *
import cv2
import argparse
from keras.utils.np_utils import to_categorical
BATCH_SIZE = 32
NB_EPOCHS = 50

def train(dataset_path):
        train_generator = get_batches(dataset_path+"/train", shuffle=False, batch_size=BATCH_SIZE)
        valid_generator = get_batches(dataset_path+"/valid", shuffle=False, batch_size=BATCH_SIZE)
        x_train = train_generator.classes
        x_valid = valid_generator.classes
        y_train = to_categorical(x_train)
        nb_classes = len(y_train[0])
        model = get_model(nb_classes)
        nb_train_samples = len(x_train)
        nb_valid_samples = len(x_valid)
        print(nb_train_samples)
        checkpoint_path="weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
        model.fit_generator(
                train_generator, 
                nb_train_samples,
                NB_EPOCHS,
                validation_data=valid_generator,
                nb_val_samples=nb_valid_samples,
                callbacks=[checkpoint])

def visualize_class_activation_map(model_path, img_path, output_path):
        model = load_model(model_path)
        original_img = cv2.imread(img_path, 1)
        width, height, _ = original_img.shape

        #Reshape to the network input shape (3, w, h).
        img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])
        
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = get_output_layer(model, "conv5_3")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]

        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
        for i, w in enumerate(class_weights[:, 1]):
                cam += w * conv_outputs[i, :, :]
        print "predictions", predictions
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img
        cv2.imwrite(output_path, img)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = bool, default = False, help = 'Train the network or visualize a CAM')
    parser.add_argument("--image_path", type = str, help = "Path of an image to run the network on")
    parser.add_argument("--output_path", type = str, default = "heatmap.jpg", help = "Path of an image to run the network on")
    parser.add_argument("--model_path", type = str, help = "Path of the trained model")
    parser.add_argument("--dataset_path", type = str, help = \
        'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
        http://pascal.inrialpes.fr/data/human/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()
        if args.train:
                train(args.dataset_path)
        else:
                visualize_class_activation_map(args.model_path, args.image_path, args.output_path)
