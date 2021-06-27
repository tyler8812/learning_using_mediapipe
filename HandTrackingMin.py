import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()

    # only read RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get result of hands
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
