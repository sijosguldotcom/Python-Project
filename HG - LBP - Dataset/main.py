import cv2
from WebcamCapture import WebcamCapture
from HandDetection import HandDetection
from ProcessImage import ProcessImage
from utils import collect_dataset


images, labels, label_dic = collect_dataset()
rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)


camera = WebcamCapture()
hand_detector = HandDetection()
image_processor = ProcessImage()

collector = cv2.face.StandardCollector_create()

while True:
    frame = camera.get_feed()
    hands_area = hand_detector.detect_hand(frame)

    if len(hands_area):
        hands = image_processor.normalize_image(frame, hands_area)
        hand = hands[0]
        rec_lbph.predict_collect(hand, collector)
        conf = collector.getMinDist()
        pred = collector.getMinLabel()
        threshold = 140
        print('LBPH hands -> Prediction: ' + label_dic[pred].capitalize() +
              ' Confidence: ' + str(round(conf)))

        for x, y, width, height in hands_area:
            frame = cv2.rectangle(frame, (x, y), (x + width, y + height),
                                  (0, 0, 255), 3)

            if conf < threshold:
                frame = cv2.putText(frame, label_dic[pred].capitalize(),
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                    1, cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, "Unknown",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                    1, cv2.LINE_AA)

    cv2.imshow('HAND DETECTION', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del camera
del hand_detector
del image_processor
