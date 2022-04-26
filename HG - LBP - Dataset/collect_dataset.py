import cv2
from WebcamCapture import WebcamCapture
from CaptureImage import StoreImage
from HandDetection import HandDetection


def capture_image():
    camera = WebcamCapture()
    hand_detector = HandDetection()
    image_capturer = StoreImage()
    name = input('Please input your name: ')
    counter = 0
    timer = 0

    if image_capturer.user_exists(name):
        print('User Already Exists. Try a New Name.')
        return

    while True:
        if counter < 20:
            frame = camera.get_feed()
            hands = hand_detector.detect_hand(frame)

            if len(hands):
                header = "Please Wait Till We Take Pictures"

            else:
                header = "Please Stay in Front of Camera"

            if len(hands) and timer % 10 == 1:
                image_capturer.save_image(frame, hands, name, counter)
                counter += 1

            for x, y, width, height in hands:
                frame = cv2.rectangle(frame, (x, y), (x + width, y + height),
                                      (0, 0, 255), 3)
                label = 'Hand'
                frame = cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                    1, cv2.LINE_AA)

            frame = cv2.putText(frame, header,
                                (10, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                1, cv2.LINE_AA)

            frame = cv2.putText(frame, "Total: 20",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                1, cv2.LINE_AA)

            frame = cv2.putText(frame, "Picture Taken: " + str(counter),
                                (100, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                1, cv2.LINE_AA)
            cv2.imshow('Video Capturing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            timer += 1

        else:
            break

    cv2.destroyAllWindows()
    del camera
    del image_capturer
    del hand_detector


capture_image()
