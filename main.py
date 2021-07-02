import cv2 as cv
import numpy as np

from core import Image, CameraSource, Curve
from filters import Kernel, NonLinear, Morph, EdgeDetection
from functions import concatenate






watch = Image.from_path("samples/watch.png")
img = CameraSource(0)

watch.get_copy()

tmp = img.get_copy()

while True:

    img.get_next_frame().make_grayscale().filter(EdgeDetection.canny(60,60)).show("Stream")
    

    if (cv.waitKey(5) & 0xFF == ord('q')):
        # press 'q' to quit loop
        break

print("Exited loop")
img.release_camera()

cv.waitKey(0)   # press any key to quit the program

cv.destroyAllWindows()
