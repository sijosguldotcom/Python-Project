import cv2
import os


class HandDetection():
    BASE_PATH = os.path.dirname(__file__)
    frontal_cascade_path = os.path.join(BASE_PATH,
                                        'cascades/frontal.xml')
    profile_cascade_path = os.path.join(BASE_PATH,
                                        'cascades/profile.xml')

    def __init__(self):
        self.frontal_detection = cv2.CascadeClassifier(
            self.frontal_cascade_path)
        self.profile_detection = cv2.CascadeClassifier(
            self.profile_cascade_path)

    def detect_hand(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hands = self.frontal_detection.detectMultiScale(gray,
                                                        scaleFactor=1.07,
                                                        minNeighbors=5)

        if len(hands) < 1:
            hands = self.profile_detection.detectMultiScale(gray,
                                                            scaleFactor=1.07,
                                                            minNeighbors=5)

        return hands
