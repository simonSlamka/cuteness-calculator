import cv2
import numpy as np
from utils import calculate_euclidean_distance, aspect_ratio
from landmarks_detector import LandmarksDetector

class CutenessCalculator:
    def __init__(self, predictor_path):
        self.landmarks_detector = LandmarksDetector(predictor_path)

    def calculate_eye_size(self, landmarks):
        # Assuming landmarks 36-41 and 42-47 correspond to eye landmarks
        right_eye = landmarks[36:42]
        left_eye = landmarks[42:48]
        eye_area = cv2.contourArea(np.array(right_eye)) + cv2.contourArea(np.array(left_eye))
        return eye_area

    def calculate_cheek_fullness(self, landmarks):
        jaw_width = np.linalg.norm(np.array(landmarks[3]) - np.array(landmarks[15]))
        midface_height = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[27]))
        cheek_fullness = jaw_width / midface_height
        return cheek_fullness

    def calculate_smile_width(self, landmarks):
        smile_width = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[54]))
        return smile_width

    def calculate_facial_proportions(self, landmarks):
        # This is a simplified example; you might want to refine these calculations
        forehead_height = np.linalg.norm(np.array(landmarks[19]) - np.array(landmarks[24]))
        midface_height = np.linalg.norm(np.array(landmarks[39]) - np.array(landmarks[42]))
        lower_face_height = np.linalg.norm(np.array(landmarks[31]) - np.array(landmarks[35]))
        proportions = [forehead_height, midface_height, lower_face_height]
        return proportions

    def calculate_eye_to_face_ratio(self, landmarks, eye_area):
        face_area = cv2.contourArea(np.concatenate((landmarks[0:17], landmarks[17:27])))
        ratio = eye_area / face_area
        return ratio

    def calculate_cuteness(self, image_path):
        """
        Calculate the cuteness of a face in an image.
        """
        # Get the facial landmarks from the image
        landmarks = self.landmarks_detector.get_landmarks(image_path)

        # Calculate the aspect ratio of the eyes
        left_eye = aspect_ratio(landmarks[42:48])
        right_eye = aspect_ratio(landmarks[36:42])
        avg_eye_aspect_ratio = (left_eye + right_eye) / 2.0

        eyeArea = self.calculate_eye_size(landmarks)
        cheekFullness = self.calculate_cheek_fullness(landmarks)
        smileWidth = self.calculate_smile_width(landmarks)
        facialProportions = self.calculate_facial_proportions(landmarks)
        eyeToFaceRatio = self.calculate_eye_to_face_ratio(landmarks, eyeArea)

        # Calculate the width and height of the face
        face_width = calculate_euclidean_distance(landmarks[0], landmarks[16])
        face_height = calculate_euclidean_distance(landmarks[8], landmarks[27])

        # Calculate the aspect ratio of the face
        face_aspect_ratio = face_width / face_height

        # Calculate the cuteness score based on the aspect ratios
        cuteness_score = self._calculate_score(avg_eye_aspect_ratio, face_aspect_ratio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio)

        return cuteness_score

    def _calculate_score(self, eye_aspect_ratio, face_aspect_ratio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio):
        """
        Calculate the cuteness score based on the aspect ratios.
        """

        weights = {
            "eyeWeight": 1,
            "faceWeight": 1,
            "eyeAreaWeight": 1,
            "cheekFullnessWeight": 1,
            "smileWidthWeight": 1,
            "facialProportionsWeight": 1,
            "eyeToFaceRatioWeight": 1
        }

        # Normalize the aspect ratios to a range of 0 to 1
        normalized_eye_aspect_ratio = (eye_aspect_ratio - 0.2) / (0.6 - 0.2)
        normalized_face_aspect_ratio = (face_aspect_ratio - 1.0) / (1.6 - 1.0)
        normalized_eyeArea = (eyeArea - 2000) / (10000 - 2000)
        normalized_cheekFullness = (cheekFullness - 0.4) / (1.0 - 0.4)
        normalized_smileWidth = (smileWidth - 0.0) / (0.3 - 0.0)
        normalized_facialProportions = (facialProportions[0] - 0.0) / (0.3 - 0.0)
        normalized_eyeToFaceRatio = (eyeToFaceRatio - 0.0) / (0.1 - 0.0)

        # Calculate the weighted average of the aspect ratios
        score = (
            weights["eyeWeight"] * normalized_eye_aspect_ratio +
            weights["faceWeight"] * normalized_face_aspect_ratio +
            weights["eyeAreaWeight"] * normalized_eyeArea +
            weights["cheekFullnessWeight"] * normalized_cheekFullness +
            weights["smileWidthWeight"] * normalized_smileWidth +
            weights["facialProportionsWeight"] * normalized_facialProportions +
            weights["eyeToFaceRatioWeight"] * normalized_eyeToFaceRatio)

        # Apply softmax to move the score to range [0, 1]
        score = np.exp(score) / np.sum(np.exp(score))

        return score
