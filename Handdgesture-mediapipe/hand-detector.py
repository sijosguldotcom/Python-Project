import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    image = cv2.flip(image,1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # Fungsi untuk cek tangan ketika terdeteksi
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # berjalan di setiap tangan yang tertangkap
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

    if id == 20 :
        cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Detection", image)
    if cv2.waitKey(1) == ord('a'):
        break
    cv2.waitKey(1)
