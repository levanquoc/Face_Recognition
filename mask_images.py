import cv2
import os
from mask import create_mask

from imutils import paths
path="./dataset_maskedface"
folder=(os.listdir("dataset"))

for i,namefolder in enumerate(folder):
    if namefolder in os.listdir(path):
        pass
    else:    
        try:
            os.mkdir(os.path.join(path,namefolder))
        except Exception as e:
            print(e)

#c = 0
images = list(paths.list_images("dataset"))
for i in range(len(images)):
    print("the path of the image is", images[i])
    #image = cv2.imread(images[i])
    #c = c + 1
    create_mask(images[i])
    