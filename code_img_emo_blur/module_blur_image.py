from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

def blur_image(boxes, frame, image):

    for j in range(0, len(boxes)):
        mask = Image.new('L', frame.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([(boxes[j,0], boxes[j,1]), (boxes[j,2], boxes[j,3])], fill=255)
        blurred = frame.filter(ImageFilter.GaussianBlur(52))
        frame.paste(blurred, mask=mask)

    frame.save(image)

    return