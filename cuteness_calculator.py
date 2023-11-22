import cv2
import numpy as np
from utils import calculate_euclidean_distance, aspect_ratio, resize_image, check_landmarks_presence, load_image, preprocess_image_for_landmark_detection
from landmarks_detector import LandmarksDetector
import os
import logging
import tempfile
import dlib
import math

logging.basicConfig(level=logging.INFO)

class CutenessCalculator:
    def __init__(self, predictorPath):
        self.landmarksDetector = LandmarksDetector(predictorPath)

    def calculate_feature_ranges(self, directoryPath, imgPath):
        directoryPath = os.path.abspath(directoryPath)
        minValues = np.inf * np.ones(12)
        maxValues = -np.inf * np.ones(12)

        if os.path.exists("minValues.npy") and os.path.exists("maxValues.npy"):
            minValues = np.load("/tmp/minValues.npy")
            maxValues = np.load("/tmp/maxValues.npy")
            logging.info("Loaded feature ranges from /tmp")
        else:
            i = 0
            logging.info(f"Calculating feature ranges for images in {directoryPath}")

            for filename in os.listdir(directoryPath):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                    imagePath = os.path.join(directoryPath, filename)
                    res = self.landmarksDetector.get_landmarks(imagePath)

                    if res is None:
                        logging.error(f"Could not get landmarks for image at {imagePath}")
                        continue

                    landmarks, img = res

                    if not check_landmarks_presence(landmarks=landmarks):
                        logging.error(f"Landmarks missing! Skipping image at {imagePath}")
                        continue

                    i += 1

                    eyeAspectRatio = aspect_ratio(np.concatenate((landmarks[36:42], landmarks[42:48]))) / 2.0
                    faceAspectRatio = calculate_euclidean_distance(landmarks[0], landmarks[16]) / calculate_euclidean_distance(landmarks[8], landmarks[27])
                    eyeArea = self.calculate_eye_size(landmarks)
                    cheekFullness = self.calculate_cheek_fullness(landmarks)
                    smileWidth = self.calculate_smile_width(landmarks)
                    facialProportions = np.mean(self.calculate_facial_proportions(landmarks))
                    eyeToFaceRatio = self.calculate_eye_to_face_ratio(landmarks, eyeArea)

                    noseSize = self.calculate_nose_size(landmarks)
                    eyebrowShape = self.calculate_eyebrow_shape(landmarks)
                    lipFullness = self.calculate_lip_fullness(landmarks)
                    skinSmoothness = self.calculate_skin_smoothness(imagePath)
                    symmetry = self.calculate_symmetry(landmarks)

                    features = np.array([eyeAspectRatio, faceAspectRatio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio, noseSize, eyebrowShape, lipFullness, skinSmoothness, symmetry])
                    minValues = np.minimum(minValues, features)
                    maxValues = np.maximum(maxValues, features)

                    logging.info(f"Done with image {imagePath}")

            np.save("/tmp/minValues.npy", minValues)
            np.save("/tmp/maxValues.npy", maxValues)

            logging.info(f"Calculated feature intervals for {i} images")

        return minValues, maxValues

    def calculate_nose_size(self, landmarks):
        noseTip = landmarks[33]
        leftNostril = landmarks[31]
        rightNostril = landmarks[35]
        noseSize = cv2.contourArea(np.array([noseTip, leftNostril, rightNostril]))
        return noseSize

    def calculate_eyebrow_shape(self, landmarks):
        leftEyebrow = landmarks[17:22]
        rightEyebrow = landmarks[22:27]
        eyebrowArea = cv2.contourArea(np.array(leftEyebrow)) + cv2.contourArea(np.array(rightEyebrow))
        return eyebrowArea

    def calculate_lip_fullness(self, landmarks):
        lips = landmarks[48:60]
        lipArea = cv2.contourArea(np.array(lips))
        return lipArea

    def calculate_skin_smoothness(self, imagePath):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        smoothness = cv2.Laplacian(blur, cv2.CV_64F).var()
        return smoothness

    def calculate_symmetry(self, landmarks):
        leftFace = landmarks[0:16]
        rightFace = landmarks[16:27]
        leftArea = cv2.contourArea(np.array(leftFace))
        rightArea = cv2.contourArea(np.array(rightFace))
        symmetry = abs(leftArea - rightArea)
        return symmetry

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

    def draw_features(self, landmarks, image):
        if image is not None:
            image = resize_image(image)
            overlay = image.copy()

            for (x, y) in landmarks:
                cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

            cv2.line(image, tuple(landmarks[36]), tuple(landmarks[45]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmarks[8]), tuple(landmarks[27]), (255, 0, 0), 2)

            cv2.polylines(image, [np.array([landmarks[33], landmarks[31], landmarks[35]])], True, (0, 255, 0), 2)

            cv2.polylines(image, [np.array(landmarks[17:22])], False, (0, 255, 255), 2)
            cv2.polylines(image, [np.array(landmarks[22:27])], False, (0, 255, 255), 2)

            cv2.polylines(image, [np.array(landmarks[48:60])], True, (255, 0, 255), 2)

            # faceCenterX = (landmarks[36][0] + landmarks[45][0]) // 2
            # cv2.line(image, (faceCenterX, landmarks[27][1]), (faceCenterX, landmarks[8][1]), (255, 255, 0), 2)

            # alpha = 0.75
            # output = cv2.addWeighted(image, alpha, image, 1 - alpha, 0)

            cv2.namedWindow("out", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("out", image)
            cv2.resizeWindow("out", 1200, 1200)
            cv2.waitKey(0)


    def calculate_cuteness(self, imagePath, minValues, maxValues):
        """
        Calculate the cuteness of a face in an image.
        """
        landmarks, image = self.landmarksDetector.get_landmarks(imagePath)

        if not check_landmarks_presence(landmarks):
            logging.error(f"Missing essential landmarks in image at {imagePath}")
            return None

        self.draw_features(landmarks, image)

        leftEye = aspect_ratio(landmarks[42:48])
        rightEye = aspect_ratio(landmarks[36:42])
        avgEyeAspectRatio = (leftEye + rightEye) / 2.0

        eyeArea = self.calculate_eye_size(landmarks)
        cheekFullness = self.calculate_cheek_fullness(landmarks)
        smileWidth = self.calculate_smile_width(landmarks)
        facialProportions = self.calculate_facial_proportions(landmarks)
        eyeToFaceRatio = self.calculate_eye_to_face_ratio(landmarks, eyeArea)

        noseSize = self.calculate_nose_size(landmarks)
        eyebrowShape = self.calculate_eyebrow_shape(landmarks)
        lipFullness = self.calculate_lip_fullness(landmarks)
        skinSmoothness = self.calculate_skin_smoothness(imagePath)
        symmetry = self.calculate_symmetry(landmarks)

        faceWidth = calculate_euclidean_distance(landmarks[0], landmarks[16])
        faceHeight = calculate_euclidean_distance(landmarks[8], landmarks[27])

        faceAspectRatio = faceWidth / faceHeight

        cutenessScore = self._calculate_score(avgEyeAspectRatio, faceAspectRatio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio, noseSize, eyebrowShape, lipFullness, skinSmoothness, symmetry, minValues, maxValues)

        return cutenessScore

    def _calculate_score(self, avgEyeAspectRatio, faceAspectRatio, eyeArea, cheekFullness, smileWidth, facialProportions, eyeToFaceRatio, noseSize, eyebrowShape, lipFullness, skinSmoothness, symmetry, minValues, maxValues):
        """
        Calculate the cuteness score based on the aspect ratios.
        """

        weights = np.array([0.15, 0.05, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05])

        normalizedEyeAspectRatio = (avgEyeAspectRatio - minValues[0]) / (maxValues[0] - minValues[0])
        normalizedFaceAspectRatio = (faceAspectRatio - minValues[1]) / (maxValues[1] - minValues[1])
        normalizedEyeArea = (eyeArea - minValues[2]) / (maxValues[2] - minValues[2])
        normalizedCheekFullness = (cheekFullness - minValues[3]) / (maxValues[3] - minValues[3])
        normalizedSmileWidth = (smileWidth - minValues[4]) / (maxValues[4] - minValues[4])
        normalizedFacialProportions = np.mean([(fp - minValues[5]) / (maxValues[5] - minValues[5]) for fp in facialProportions])
        normalizedEyeToFaceRatio = (eyeToFaceRatio - minValues[6]) / (maxValues[6] - minValues[6])
        normalizedNoseSize = (noseSize - minValues[7]) / (maxValues[7] - minValues[7])
        normalizedEyebrowShape = (eyebrowShape - minValues[8]) / (maxValues[8] - minValues[8])
        normalizedLipFullness = (lipFullness - minValues[9]) / (maxValues[9] - minValues[9])
        EPSILON = 1e-7
        normalizedSkinSmoothness = (skinSmoothness - minValues[10]) / (maxValues[10] - minValues[10] + EPSILON)
        normalizedSymmetry = (symmetry - minValues[11]) / (maxValues[11] - minValues[11])

        values = np.array([normalizedEyeAspectRatio, normalizedFaceAspectRatio, normalizedEyeArea, normalizedCheekFullness, normalizedSmileWidth, normalizedFacialProportions, normalizedEyeToFaceRatio, normalizedNoseSize, normalizedEyebrowShape, normalizedLipFullness, normalizedSkinSmoothness, normalizedSymmetry])
        values = np.abs(values)
        weightedValues = weights * values

        score = np.sum(weightedValues)

        return score