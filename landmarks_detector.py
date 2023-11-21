import cv2
import dlib
import numpy as np
from utils import load_image, convert_to_gray, resize_image
import logging

class LandmarksDetector:
    def __init__(self, predictorPath):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictorPath)

    def get_landmarks(self, imagePath):
        """
        Get the facial landmarks from an image.
        """
        image = load_image(imagePath)
        image = resize_image(image)
        gray = convert_to_gray(image)

        landmarks = self.detector(gray, 1)

        # if len(landmarks) == 0:
        #     raise ValueError('No landmarks detected in {}'.format(imagePath))

        # For each detected face, find the landmark.
        for _, rect in enumerate(landmarks):
            shape = self.predictor(gray, rect)
            shape = self._shape_to_np(shape)

            # Draw bounding box for each face detected
            x = rect.left()
            y = rect.top()
            w = rect.width()
            h = rect.height()
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 1)

            # Estimate forehead
            forehead_height = int(0.25 * h)
            cv2.rectangle(image, (x, y - forehead_height), (x + w, y), (0, 255, 255), 1)

            for (x, y) in shape:
                x = int(x * image.shape[1] / gray.shape[1])
                y = int(y * image.shape[0] / gray.shape[0])

                if x < rect.left() or x > rect.right() or y < rect.top() or y > rect.bottom():
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                    logging.warning("Landmark outside of bounding box at coords ({}, {})".format(x, y))
                else:
                    cv2.circle(image, (x, y), 1, (255, 0, 255), -1)
            cv2.namedWindow("out", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("out", image)
            cv2.resizeWindow("out", 1024, 1024)
            cv2.waitKey(0)

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
