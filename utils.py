import cv2
import numpy as np
from scipy.spatial import distance as dist

def load_image(imagePath):
    """
    Load an image from a file path.
    """
    image = cv2.imread(imagePath)
    return image

def convert_to_gray(image):
    """
    Convert a given image to grayscale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def resize_image(image, width=224):
    """
    Resize the image to a given width.
    """
    r = float(width) / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def calculate_euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return dist.euclidean(point1, point2)

def aspect_ratio(eye):
    """
    Compute the euclidean distances between the two sets of
    vertical eye landmarks (x, y)-coordinates, then compute the
    euclidean distance between the horizontal eye landmark
    (x, y)-coordinates, and finally compute the eye aspect ratio.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear