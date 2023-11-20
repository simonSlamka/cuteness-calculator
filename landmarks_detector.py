import cv2
import dlib
import numpy as np
from utils import load_image, convert_to_gray, resize_image

class LandmarksDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self, image_path):
        """
        Get the facial landmarks from an image.
        """
        image = load_image(image_path)
        image = resize_image(image)
        gray = convert_to_gray(image)

        # Detect faces in the grayscale image
        rects = self.detector(gray, 1)

        if len(rects) == 0:
            raise ValueError('No landmarks detected in {}'.format(image_path))

        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = self._shape_to_np(shape)

            return shape

    def _shape_to_np(self, shape, dtype="int"):
        """
        Convert a shape object to a numpy array.
        """
        coords = np.zeros((68, 2), dtype=dtype)

        # Loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords
