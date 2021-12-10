from cv2 import cv2
import numpy as np
import HandTracking as ht
import time
import autopy

camW, camH = 640, 480
frame_r = 150  # frame reduction
smoothening = 5

pTime = 0
plo_x, plo_y = 0, 0
w_screen, h_screen = autopy.screen.size()

clo_x, clo_y = 0, 0

cap = cv2.VideoCapture(1)
cap.set(3, camW)
cap.set(4, camH)
detector = ht.HandDetector(max_hands=1)

print(w_screen, h_screen)
while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list, b_box = detector.find_position(img)
    # 2. Get the tip of the index and middle fingers
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        # 3. Check which fingers are up
        fingers = detector.fingers_up()
        # print(fingers)
        cv2.rectangle(img, (frame_r, frame_r), (camW - frame_r, camH - frame_r), (255, 0, 255), 2)

        # 4. Only index finger => moving mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 5. Convert coordinates
            x3 = np.interp(x1, (frame_r, camW - frame_r), (0, w_screen))
            y3 = np.interp(y1, (frame_r, camH - frame_r), (0, h_screen))
            print((x1, y1), (x2, y2), (x3, y3))
            # 6. Smoothen values
            clo_x = plo_x + (x3 - plo_x) / smoothening
            clo_y = plo_y + (y3 - plo_y) / smoothening

            # 7. Move
            try:
                autopy.mouse.move(w_screen - clo_x, clo_y)
                plo_x, plo_y = clo_x, clo_y
            except:
                print((w_screen - x3, y3), " out of range")
            cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
        # 8. Both Index and middle fingers are up => clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, linf = detector.find_distance(8, 4, img)
            # print(length)

            # 10. Click mouse if distance short
            if length < 40:
                cv2.circle(img, (linf[4], linf[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(3)
