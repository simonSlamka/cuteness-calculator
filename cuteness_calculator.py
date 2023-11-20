import cv2
import numpy as np
from utils import calculate_euclidean_distance, aspect_ratio
from landmarks_detector import LandmarksDetector

class CutenessCalculator:
    def __init__(self, predictorPath):
        self.landmarksDetector = LandmarksDetector(predictorPath)

    def calculate_eye_size(self, landmarks):
        rightEye = landmarks[36:42]
        leftEye = landmarks[42:48]
        eyeArea = cv2.contourArea(np.array(rightEye)) + cv2.contourArea(np.array(leftEye))
        return eyeArea

    def calculate_cheek_fullness(self, landmarks):
        jawWidth = np.linalg.norm(np.array(landmarks[3]) - np.array(landmarks[15]))
        midfaceHeight = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[27]))
        cheekFullness = jawWidth / midfaceHeight
        return cheekFullness

    def calculate_smile_width(self, landmarks):
        smileWidth = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[54]))
        return smileWidth

    def calculate_facial_proportions(self, landmarks):
        foreheadHeight = np.linalg.norm(np.array(landmarks[19]) - np.array(landmarks[24]))
        midfaceHeight = np.linalg.norm(np.array(landmarks[39]) - np.array(landmarks[42]))
        lowerFaceHeight = np.linalg.norm(np.array(landmarks[31]) - np.array(landmarks[35]))
        proportions = [foreheadHeight, midfaceHeight, lowerFaceHeight]
        return proportions

    def calculate_eye_to_face_ratio(self, landmarks, eyeArea):
        faceArea = cv2.contourArea(np.concatenate((landmarks[0:17], landmarks[17:27])))
        ratio = eyeArea / faceArea
        return ratio

    def calculate_cuteness(self, imagePath):
        """
        Calculate the cuteness of a face in an image.
        """
        landmarks = self.landmarksDetector.get_landmarks(imagePath)

        leftEye = aspect_ratio(landmarks[42:48])
        rightEye = aspect_ratio(landmarks[36:42])
        avgEyeAspectRatio = (leftEye + rightEye) / 2.0

        eyeArea = self.calculate_eye_size(landmarks)
        cheekFullness = self.calculate_cheek_fullness(landmarks)
        smileWidth = self.calculate_smile_width(landmarks)
        facialProportions = self.calculate_facial_proportions(landmarks)
        eyeToFaceRatio = self.calculate_eye_to_face_ratio(landmarks, eyeArea)

        faceWidth = calculate_euclidean_distance(landmarks[0], landmarks[16])
        faceHeight = calculate_euclidean_distance(landmarks[8], landmarks[27])

        faceAspectRatio = faceWidth / faceHeight

        cutenessScore = self._calculate_score(avgEyeAspectRatio, faceAspectRatio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio)

        return cutenessScore

    def _calculate_score(self, eyeAspectRatio, faceAspectRatio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio):
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

        normalizedEyeAspectRatio = (eyeAspectRatio - 0.2) / (0.6 - 0.2)
        normalizedFaceAspectRatio = (faceAspectRatio - 1.0) / (1.6 - 1.0)
        normalizedEyeArea = (eyeArea - 2000) / (10000 - 2000)
        normalizedCheekFullness = (cheekFullness - 0.4) / (1.0 - 0.4)
        normalizedSmileWidth = (smileWidth - 0.0) / (0.3 - 0.0)
        normalizedFacialProportions = (facialProportions[0] - 0.0) / (0.3 - 0.0)
        normalizedEyeToFaceRatio = (eyeToFaceRatio - 0.0) / (0.1 - 0.0)

        score = (
            weights["eyeWeight"] * normalizedEyeAspectRatio +
            weights["faceWeight"] * normalizedFaceAspectRatio +
            weights["eyeAreaWeight"] * normalizedEyeArea +
            weights["cheekFullnessWeight"] * normalizedCheekFullness +
            weights["smileWidthWeight"] * normalizedSmileWidth +
            weights["facialProportionsWeight"] * normalizedFacialProportions +
            weights["eyeToFaceRatioWeight"] * normalizedEyeToFaceRatio)

        score = np.exp(score) / np.sum(np.exp(score))

        return score
