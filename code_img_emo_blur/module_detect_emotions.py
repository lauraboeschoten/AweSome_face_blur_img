import tensorflow as tf
import tensorflow.keras
import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

tf.keras.backend.clear_session()
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

model2 = tensorflow.keras.Model(inputs=model.input,
                                outputs=model.get_layer('dense').output)

def detect_emotions(gray, boxes):

    lastlayer = [None] * len(boxes)
    prediction = [None] * len(boxes)
    emo_class  = [None] * len(boxes)

    boxes = np.where(boxes < 0, 0, boxes)

    for j in range(len(boxes)):
        x = boxes[j,0]
        y = boxes[j,1]
        w = boxes[j,2]
        h = boxes[j,3]
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        lastlayer[j]  = model2.predict(cropped_img)
        prediction[j] = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction[j]))
        emo_class[j] = emotion_dict[maxindex]

    return(lastlayer,
           prediction,
           emo_class)

