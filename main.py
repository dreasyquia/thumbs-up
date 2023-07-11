import cv2
from cvzone.HandTrackingModule import HandDetector
from playsound import playsound
import time
import numpy as np
import os

cap = cv2.VideoCapture(0)

# load the overlay file
overlay = cv2.imread('resources/wonwoo-overlay.png')

# detect which pixels in the overlay have something in them
# and make a binary mask out of it
overlayMask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
res, overlayMask = cv2.threshold(overlayMask, 1, 1, cv2.THRESH_BINARY_INV)

# expand the mask from 1-channel to 3-channel
h, w = overlayMask.shape
overlayMask = np.repeat(overlayMask, 3).reshape((h, w, 3))

detector = HandDetector(maxHands=1)

photoNum = 0
timer = 0
startCountdown = False
takePhoto = False

os.makedirs('photos', exist_ok=True)
base_path = os.path.join('photos', 'wonwoo')

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    # mask out the pixels that you want to overlay
    img[36:720, 320:1280] *= overlayMask

    # put the overlay on
    img[36:720, 320:1280] += overlay

    if takePhoto:
        playsound('resources/camera-shutter.mp3')
        cv2.imwrite('{}_{}.{}'.format(base_path, photoNum, 'jpg'), img)
        photoNum += 1
        takePhoto = False

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        if fingers == [1, 0, 0, 0, 0]:
            initialTime = time.time()
            startCountdown = True

    if startCountdown:
        cv2.putText(img, "Get ready!", (430, 327), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 255, 255), 5)
        timer = time.time() - initialTime
        cv2.putText(img, str(int(timer)), (607, 459), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 255, 255), 5)

        if timer > 3:
            timer = 0
            startCountdown = False
            takePhoto = True

    cv2.imshow("ThumbsUp", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyWindow("ThumbsUp")
