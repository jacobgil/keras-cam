## Keras implementation of class activation mapping

Paper / project page: http://cnnlocalization.csail.mit.edu

Paper authors' code with Caffe / matcaffe interface: https://github.com/metalbubble/CAM


Blog post on this repository: http://jacobcv.blogspot.com/2016/08/class-activation-maps-in-keras.html 

Checkpoint with person/not person weights: https://drive.google.com/open?id=0B1l5JSkBbENBdk95ZW1DOUhqQUE

![enter image description here](https://raw.githubusercontent.com/jacobgil/keras-cam/master/examples/mona_lisa.jpg)


This project implements class activation maps with Keras.

Class activation maps are a simple technique to get the image regions relevant to a certain class.

This was fined tuned on VGG16 with images from here: 
http://pascal.inrialpes.fr/data/human

The model in model.py is a two category classifier, used to classify person / not a person.

	python cam.py --model_path cam_checkpoint.hdf5 --image_path=image.jpg

    usage: cam.py [-h] [--train TRAIN] [--image_path IMAGE_PATH]
              [--output_path OUTPUT_PATH] [--model_path MODEL_PATH]
              [--dataset_path DATASET_PATH]

	optional arguments:
	  -h, --help            show this help message and exit
	  --train TRAIN         Train the network or visualize a CAM
	  --image_path IMAGE_PATH
	                        Path of an image to run the network on
	  --output_path OUTPUT_PATH
	                        Path of an image to run the network on
	  --model_path MODEL_PATH
	                        Path of the trained model
	  --dataset_path DATASET_PATH
	                        Path to image dataset. Should have pos/neg folders,
	                        like in the inria person dataset.
	                        http://pascal.inrialpes.fr/data/human/

