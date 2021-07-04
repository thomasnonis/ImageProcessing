import cv2 as cv
import numpy as np

from core import Image, CameraSource
from filters import Kernel, NonLinear, EdgeDetection

def example_1():
    # Imports image, converts it to grayscale, computes its negative and shows the result
    img = Image.from_path("samples/lena.png").make_grayscale().negative().show("Negative")
    cv.waitKey(0)

def example_2():
    # Imports image as grayscale, filters it with a non separated Prewitt kernel and shows the result
    img = Image.from_path("samples/baboon.png", cv.IMREAD_GRAYSCALE).filter(Kernel.prewitt(), force_inseparable=True).show("Filtered")
    cv.waitKey(0)

def example_3():
    # Opens a stream from the first camera source on the pc and, until the user presses 'q', shows the real time edge detection using the Canny algorithm
    img = CameraSource(0)

    while True:

        img.get_next_frame().make_grayscale().filter(EdgeDetection.canny(60,60)).show("Stream")
        

        if (cv.waitKey(5) & 0xFF == ord('q')):
            # press 'q' to quit loop
            break

    img.release_camera()
    cv.destroyAllWindows()

def example_4():
    # Imports image, filters it with a median filter, writes the intermediate result on a file, detects the edges with the Canny algorighm and shows the final result
    img = Image.from_path("samples/watch.png").filter(NonLinear.median(5)).write("example_4.jpg").filter(EdgeDetection.canny(80, 80)).show("Edges")
    cv.waitKey(0)