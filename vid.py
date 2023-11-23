import cv2
from cuteness_calculator import CutenessCalculator
from utils import resize_image
from tqdm import tqdm

class Video:
    def __init__(self, videoFile, predictorPath):
        self.videoFile = videoFile
        self.cutenessCalculator = CutenessCalculator(predictorPath)

    def process_video(self, outputPath):
        videoStream = cv2.VideoCapture(self.videoFile)
        totalFrames = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))
        progressBar = tqdm(total=totalFrames, desc="Processing video", ncols=80)

        _, singleFrame = videoStream.read()

        singleFrame = resize_image(singleFrame)

        frameHeight, frameWidth, _ = singleFrame.shape[:3]
        fps = int(videoStream.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        outputPath = outputPath.replace('.mp4', '.mkv')
        out = cv2.VideoWriter(outputPath, fourcc, fps, (frameWidth, frameHeight))

        try:
            while(videoStream.isOpened()):
                ret, frame = videoStream.read()
                if ret is True:
                    modFrame = frame
                    res = self.cutenessCalculator.landmarksDetector.get_landmarks(image=frame)
                    if res is not None:
                        landmarks, frame = res
                        print(f"Landmark count: {len(landmarks)}")
                        if landmarks is not None:
                            modFrame = self.cutenessCalculator.draw_features(landmarks, frame)
                    out.write(modFrame)
                    cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
                    cv2.imshow("frame", modFrame)
                    cv2.resizeWindow("frame", 1200, 1200)
                    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                        break
                else:
                    break
                progressBar.update()
                if progressBar.n == totalFrames:
                    break
        finally:
            videoStream.release()
            out.release()
            cv2.destroyAllWindows()
            progressBar.close()


if __name__ == "__main__":
    video = Video("emi.mp4", "shape_predictor_68_face_landmarks.dat")
    video.process_video("emi_superimposed.mp4")
