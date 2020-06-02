import cv2
from facenet_pytorch import MTCNN
from PIL import Image

mtcnn = MTCNN(margin = 20,
              keep_all = True,
              post_process = False)

def detect_faces(image):

    #img   = cv2.imread(image)
    img = image
    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = Image.fromarray(color)

    boxes, probs, landmarks = mtcnn.detect(frame,
                                           landmarks = True)

    if boxes is None:
        boxes = "No faces in picture"
    else:
        boxes = boxes.astype(int)

    return(color,
           gray,
           frame,
           boxes)