import cv2
import dlib
import numpy as np
from utils import load_image, resize_image, preprocess_image_for_landmark_detection, calculate_angle, draw_rotated_rectangle, check_landmarks_presence
import logging
from mtcnn import MTCNN

logging.basicConfig(level=logging.INFO)

class LandmarksDetector:
    def __init__(self, predictorPath):
        self.detector = dlib.get_frontal_face_detector()
        self.mtcnn = MTCNN()
        self.predictor = dlib.shape_predictor(predictorPath)

    def get_landmarks(self, imagePath):
        """
        Get the facial landmarks from an image.
        """
        image = load_image(imagePath)
        preprocessed = preprocess_image_for_landmark_detection(image=image)
        preprocessed = resize_image(preprocessed)
        image = resize_image(image)

        faces = self.detector(preprocessed, 1)

        # if faces is None or len(faces) == 0:
        #     faces = self.mtcnn.detect_faces(preprocessed)

        for _, rect in enumerate(faces):
            shape = self.predictor(preprocessed, rect)
            shape = self._shape_to_np(shape)

            x = rect.left()
            y = rect.top()
            w = rect.width()
            h = rect.height()

            eye1 = shape[36]
            eye2 = shape[45]
            angle = calculate_angle(eye1, eye2)

            center = (x + w // 2, y + h // 2)
            draw_rotated_rectangle(image, center, w, h, angle)

            foreheadHeight = int(0.25 * h)
            foreheadCenter = (center[0], y - foreheadHeight // 2)
            draw_rotated_rectangle(image, foreheadCenter, w, foreheadHeight, angle, color=(0, 255, 0))

            for i, (x, y) in enumerate(shape):
                x = int(x * image.shape[1] / preprocessed.shape[1])
                y = int(y * image.shape[0] / preprocessed.shape[0])

                if x < rect.left() or x > rect.right() or y < rect.top() or y > rect.bottom():
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                    logging.warning("Landmark outside of bounding box at coords ({}, {})".format(x, y))
                    # cv2.putText(image, "Out of bounds", (x - 45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
                else:
                    # Highlight the eyes with a different color
                    if i in range(36, 48):
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                    else:
                        cv2.circle(image, (x, y), 1, (255, 0, 255), -1)

            if not check_landmarks_presence(shape):
                logging.error("Not all landmarks were detected in image {}".format(imagePath))
                return None, image

            return shape, image

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
