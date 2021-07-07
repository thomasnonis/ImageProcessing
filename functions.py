'''
    Author: Thomas Nonis
'''

import cv2 as cv
import numpy as np

from core import Image, Curve

# Functions in this library should not modify the original image, but create a copy and return the edited copy.
# To edit the original image please use the Image methods



def contrast_curve(img: Image, curve: Curve):
    return img.get_copy().contrast_curve(curve)

def threshold(img: Image, threshold):
    return img.get_copy().threshold(threshold)

def negative(img: Image):
    return img.get_copy().negative()

def concatenate(img1: Image, img2: Image, axis=1):
    return Image(np.concatenate((img1.img, img2.img), axis))
