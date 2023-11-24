import cv2
import numpy as np
from scipy.spatial import distance as dist
import logging

def load_image(imagePath):
    """
    Load an image from a file path.
    """
    image = cv2.imread(imagePath)
    return image

def calculate_angle(point1, point2):
    deltaY = point2[1] - point1[1]
    deltaX = point2[0] - point1[0]
    return np.degrees(np.arctan2(deltaY, deltaX))

def convert_to_gray(image):
    """
    Convert a given image to grayscale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def preprocess_image_for_landmark_detection(image):
    """
    Preprocess a full-color image for face landmark detection.
    """
    img = cv2.convertScaleAbs(image, alpha=1.05, beta=20)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:,:,1] = img[:,:,1]*1.1
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.5)

    # img = convert_to_gray(img)

    # img = cv2.equalizeHist(img)

    # sobelX = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
    # sobelY = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)
    # img = cv2.addWeighted(sobelX, 0.1, sobelY, 0.75, 0)

    # img = cv2.GaussianBlur(img, (11, 11), 0)

    # img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.75)

    # img = convert_to_gray(img)

    # img = cv2.equalizeHist(img)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # img = cv2.addWeighted(img, 1.65, cv2.GaussianBlur(img, (3,3), 0), -0.5, 0, img)

    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # img = cv2.Canny(img, 25, 200)

    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)

    # cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
    # cv2.imshow("Threshold", img)
    # cv2.resizeWindow("Threshold", 1200, 1200)
    # cv2.waitKey(0)

    return img

def check_landmarks_presence(landmarks):
    requiredLandmarks = [33, 31, 35, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 27, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

    if landmarks is None:
        logging.error("`landmarks` is None!!!")
        return False

    if len(landmarks) < max(requiredLandmarks) + 1:
        return False

    return True

def resize_image(image, width=850):
    """
    Resize the image to a given width.
    """
    if image is not None:
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

def draw_rotated_rectangle(image, center, width, height, angle, color=(255, 255, 0), thickness=1):
    _angle = np.deg2rad(angle)
    half_width = width / 2
    half_height = height / 2

    corners = np.array([
        [half_width, half_height],
        [half_width, -half_height],
        [-half_width, -half_height],
        [-half_width, half_height]
    ])

    rotation_matrix = np.array([
        [np.cos(_angle), -np.sin(_angle)],
        [np.sin(_angle), np.cos(_angle)]
    ])

    rotated_corners = np.dot(corners, rotation_matrix.T)
    rotated_corners += center

    cv2.polylines(image, [rotated_corners.astype(int)], isClosed=True, color=color, thickness=thickness)