import os
import pandas as pd
import numpy as np
import json
import pickle

os.chdir('C:/code_img_emo_blur')

import module_detect_faces as mdf
import module_detect_emotions as mde
#import module_save_emotions as mse
import module_blur_image as mbi

# 0. make list with images
directory = r"C:\datadownload"

path_list = [os.path.join(dirpath, filename) for dirpath, _,
                                                 filenames in os.walk(directory) for filename in filenames if
             filename.endswith('.jpg')]

# 1. Face detect

for i in range(len(path_list)):

    # function that loads image and detects faces, returns boxes file
    color, gray, frame, boxes = mdf.detect_faces(path_list[i])

    if boxes == 'No faces in picture':
        continue

    # 2. read emotions on detected faces
    lastlayer, prediction, emo_class = mde.detect_emotions(gray, boxes)


    # 3. make file with face location and emotion results

    emo_list = [boxes, prediction, emo_class, lastlayer]

    #Save file to disk with pickle
    with open(path_list[i][:-4] + '.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(emo_list, filehandle)

    ##Load file from disk to workspace
    #with open('emo_list_file.data', 'rb') as filehandle:
    #    # read the data as binary data stream
    #    emo_list_load = pickle.load(filehandle)

    # 4. blur image and save

    mbi.blur_image(boxes, frame, path_list[i])