import os
import pandas as pd
import cv2
import glob
import numpy as np

os.chdir('C:/Users/F112974/surfdrive/Onderzoek/AweSome/deduce_instagram_05_2020_3/code_vid_emo_blur')

import module_detect_faces as mdf
import module_detect_emotions as mde
import module_save_emotions as mse
import module_blur_video as mbv

# 0. make list with images
directory = r"C:\Users\F112974\surfdrive\Onderzoek\AweSome\deduce_instagram_05_2020_3\datadownload"

path_list = [os.path.join(dirpath, filename) for dirpath, _,
                                                 filenames in os.walk(directory) for filename in filenames if
             filename.endswith('.mp4')]

# 1. Face detect

for i in range(len(path_list)):

    cap = cv2.VideoCapture(path_list[i])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    img_array = []

    for g in range(total_frames):
        cap.set(1, g - 1);
        success = cap.grab()
        ret, image = cap.retrieve()
        if ret == True:
            color, gray, frame, boxes = mdf.detect_faces(image)
            if boxes == 'No faces in picture':
                img_array.append(frame)

            else:
                # detect emotions
                lastlayer, prediction, emo_class = mde.detect_emotions(gray, boxes)

                # save emotions
                mse.save_emotions(boxes, lastlayer, prediction, emo_class, path_list[i], g)

                # save blurred video
                img_array.append(mbv.blur_image(boxes, frame))


            height, width, layers = image.shape
            size = (width, height)

    out = cv2.VideoWriter(path_list[i][:-4] + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for f in range(len(img_array)):
        cvimage = np.array(img_array[f])
        out.write(cvimage)

    out.release()

