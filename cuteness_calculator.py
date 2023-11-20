import cv2
import numpy as np
from utils import calculate_euclidean_distance, aspect_ratio
from landmarks_detector import LandmarksDetector
import os
import logging

logging.basicConfig(level=logging.INFO)

class CutenessCalculator:
    def __init__(self, predictorPath):
        self.landmarksDetector = LandmarksDetector(predictorPath)

    def calculate_feature_ranges(self, directoryPath):
        directoryPath = os.path.abspath(directoryPath)
        minValues = np.inf * np.ones(7)
        maxValues = -np.inf * np.ones(7)

        if os.path.exists('/tmp/minValues.npy') and os.path.exists("/tmp/maxValues.npy"):
            minValues = np.load("/tmp/minValues.npy")
            maxValues = np.load("/tmp/maxValues.npy")
            logging.info("Loaded feature ranges from /tmp")
        else:
            i = 0
            logging.info(f"Calculating feature ranges for images in {directoryPath}")

            for filename in os.listdir(directoryPath):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    imagePath = os.path.join(directoryPath, filename)
                    landmarks = self.landmarksDetector.get_landmarks(imagePath)

                    if landmarks is None:
                        continue

                    i += 1

                    eyeAspectRatio = aspect_ratio(np.concatenate((landmarks[36:42], landmarks[42:48]))) / 2.0
                    faceAspectRatio = calculate_euclidean_distance(landmarks[0], landmarks[16]) / calculate_euclidean_distance(landmarks[8], landmarks[27])
                    eyeArea = self.calculate_eye_size(landmarks)
                    cheekFullness = self.calculate_cheek_fullness(landmarks)
                    smileWidth = self.calculate_smile_width(landmarks)
                    facialProportions = np.mean(self.calculate_facial_proportions(landmarks))
                    eyeToFaceRatio = self.calculate_eye_to_face_ratio(landmarks, eyeArea)

                    features = np.array([eyeAspectRatio, faceAspectRatio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio])
                    minValues = np.minimum(minValues, features)
                    maxValues = np.maximum(maxValues, features)

            np.save("/tmp/minValues.npy", minValues)
            np.save("/tmp/maxValues.npy", maxValues)

            logging.info(f"Calculated feature intervals for {i} images")

        return minValues, maxValues

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

    def calculate_cuteness(self, imagePath, minValues, maxValues):
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

        cutenessScore = self._calculate_score(avgEyeAspectRatio, faceAspectRatio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio, minValues, maxValues)

        return cutenessScore

    def _calculate_score(self, eyeAspectRatio, faceAspectRatio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio, minValues, maxValues):
        """
        Calculate the cuteness score based on the aspect ratios.
        """

        weights = np.array([0.25, 0.05, 0.25, 0.10, 0.15, 0.10, 0.10])

        normalizedEyeAspectRatio = (eyeAspectRatio - minValues[0]) / (maxValues[0] - minValues[0])
        normalizedFaceAspectRatio = (faceAspectRatio - minValues[1]) / (maxValues[1] - minValues[1])
        normalizedEyeArea = (eyeArea - minValues[2]) / (maxValues[2] - minValues[2])
        normalizedCheekFullness = (cheekFullness - minValues[3]) / (maxValues[3] - minValues[3])
        normalizedSmileWidth = (smileWidth - minValues[4]) / (maxValues[4] - minValues[4])
        normalizedFacialProportions = np.mean([(fp - minValues[5]) / (maxValues[5] - minValues[5]) for fp in facialProportions])
        normalizedEyeToFaceRatio = (eyeToFaceRatio - minValues[6]) / (maxValues[6] - minValues[6])

        values = np.array([normalizedEyeAspectRatio, normalizedFaceAspectRatio, normalizedEyeArea, normalizedCheekFullness, normalizedSmileWidth, normalizedFacialProportions, normalizedEyeToFaceRatio])
        values = np.abs(values)
        weightedValues = weights * values

        score = np.sum(weightedValues)

        return score