from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

def blur_image(boxes, frame):

    for j in range(0, len(boxes)):
        mask = Image.new('L', frame.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([(boxes[j,0], boxes[j,1]), (boxes[j,2], boxes[j,3])], fill=255)
        blurred = frame.filter(ImageFilter.GaussianBlur(52))
        frame.paste(blurred, mask=mask)
        #color2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #frame.save(image[:-4] + ".jpg")

    return(frame)