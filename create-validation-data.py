import os

TRAIN_DIR = "./data/train"
VALID_DIR = "./data/valid"


classes = os.listdir(TRAIN_DIR)

def percentage(percent, whole):
  return (percent * whole) / 100.0

if not os.path.exists(VALID_DIR):
    os.mkdir(VALID_DIR)

for c in classes:
    image_dir = "{}/{}/".format(TRAIN_DIR, c)
    images = os.listdir(image_dir)
    num_of_images_to_move = int(percentage(20, len(images)))

    valid_class_dir = VALID_DIR+"/"+c
    train_class_dir = TRAIN_DIR+"/"+c

    if not os.path.exists(valid_class_dir):
        os.mkdir(valid_class_dir)

    for image in images[-num_of_images_to_move:]:
        new_file_name = valid_class_dir+"/"+image
        old_file_name = train_class_dir+"/"+image
        print('moving', old_file_name, 'to', new_file_name)
        os.rename(old_file_name, new_file_name)
