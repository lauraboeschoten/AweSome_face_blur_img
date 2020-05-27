import os
import pandas as pd

os.chdir('C:/Users/F112974/surfdrive/Onderzoek/AweSome/deduce_instagram_05_2020_3/code_img_emo_blur')

import module_detect_faces as mdf
import module_detect_emotions as mde
import module_save_emotions as mse
import module_blur_image as mbi

# 0. make list with images
directory = r"C:\Users\F112974\surfdrive\Onderzoek\AweSome\deduce_instagram_05_2020_3\datadownload"

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
    prediction, emo_class = mde.detect_emotions(gray, boxes)

    # 3. make file with face location and emotion results

    mse.save_emotions(boxes, prediction, emo_class, path_list[i])

    # 4. blur image and save

    mbi.blur_image(boxes, frame, path_list[i])