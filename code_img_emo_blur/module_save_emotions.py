import pandas as pd
import numpy as np

def save_emotions(boxes, prediction, emo_class, image):

    emo_class = pd.DataFrame(emo_class, columns = ["classification"])
    prediction = np.array(prediction)
    m, n, r = prediction.shape
    out_arr = np.column_stack((np.repeat(np.arange(m), n), prediction.reshape(m * n, -1)))

    prediction = pd.DataFrame(out_arr)
    boxes = pd.DataFrame(np.array(boxes))
    #boxes = pd.DataFrame(boxes)

    df_emo_class = pd.concat([boxes, prediction.iloc[:, 1:8], emo_class], axis=1)
    df_emo_class.columns = ['face_x', 'face_y', 'face_w', 'face_h',
                            'pr_angry', 'pr_disgusted', 'pr_fearful', 'pr_happy', 'pr_neutral', 'pr_sad',
                            'pr_surprised',
                            'classification']

    df_emo_class.to_csv(image[:-4] + ".csv")