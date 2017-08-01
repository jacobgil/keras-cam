import magic
from subprocess import call
import os

DATA_DIR = "./dataset"

classes = os.listdir(DATA_DIR)

for c in classes:
    image_dir = "{}/{}/".format(DATA_DIR, c)
    images = os.listdir(image_dir)
    
    call(["mogrify", "-format", "jpeg", "{}/*.png".format(image_dir)])

    for image in images:
        file_name = image_dir+image
        mime = magic.from_file(file_name, mime=True)
        if mime != "image/jpeg":
            # print('removing', file_name)
            os.remove(file_name)
