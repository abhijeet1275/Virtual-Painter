from turtle import delay
import math
import cv2
import mediapipe as mp
import time
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

folderPath = "../header"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (0,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1275)
cap.set(4,720)

hands = mp_hands.Hands()
pTime = 0
brushThickness=15
eraserThickness=75
xp,yp=0,0
imgCanvas= np.zeros((720,1275,3),np.uint8)
while True:
    data, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finger_coordinates = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
        for landmark in mp_hands.HandLandmark:
            finger_tip = hand_landmarks.landmark[landmark]
            x, y, z = finger_tip.x, finger_tip.y, finger_tip.z

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = image.shape
            pixel_x, pixel_y = int(x * w), int(y * h)

            # Append coordinates to the array
            finger_coordinates.append((pixel_x, pixel_y))

    image[0:157,0:1275] = header

    x1, y1 = 0, 0
    x2, y2 = 0, 0
    # print("Finger Coordinates:", finger_coordinates)
    if finger_coordinates:
        # Check if index 8 is within the range of the list
        if len(finger_coordinates) >= 9:
            x1, y1 = finger_coordinates[8]
            x2, y2 = finger_coordinates[12]

    if len(finger_coordinates)!=0:
        fingers = []

        thumb_tip_x = finger_coordinates[4][0]
        thumb_base_x = finger_coordinates[2][0]
        fingers.append(1 if abs(thumb_tip_x) > abs(thumb_base_x) else 0)  # Thumb

        index_tip_y = finger_coordinates[8][1]
        index_base_y = finger_coordinates[5][1]
        fingers.append(1 if index_tip_y > index_base_y else 0)  # Index

        middle_tip_y = finger_coordinates[12][1]
        middle_base_y = finger_coordinates[9][1]
        fingers.append(1 if middle_tip_y > middle_base_y else 0)  # Middle

        ring_tip_y = finger_coordinates[16][1]
        ring_base_y = finger_coordinates[13][1]
        fingers.append(1 if ring_tip_y > ring_base_y else 0)  # Ring

        pinky_tip_y = finger_coordinates[20][1]
        pinky_base_y = finger_coordinates[17][1]
        fingers.append(1 if pinky_tip_y > pinky_base_y else 0)  # Pinky

        if fingers.count(1) == 3:
            print("Selection Mode")
            xp, yp = 0, 0
            cv2.rectangle(image,(x1, y1-25), (x2,y2+25), drawColor, cv2.FILLED)

            if y1<145:
                if 25<x1<125:
                    header = overlayList[0]
                    drawColor=(0,0,255)
                elif 200<x1<300:
                    header = overlayList[1]
                    drawColor = (0, 255,0)
                elif 390<x1<500:
                    header = overlayList[2]
                    drawColor = (0, 255, 255)
                elif 585<x1<695:
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 760<x1<880:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
                elif 1050<x1<1230:
                    header = overlayList[5]
                    drawColor = (0, 0, 0)


                cv2.rectangle(image, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        elif fingers.count(1) == 4:
            print("Drawing Mode")

            if(xp==0 and yp==0):
                xp,yp=x1,y1
            if all(f == 0 for f in fingers):
                print("Clear Screen")
                imgCanvas = np.zeros((720, 1275, 3), np.uint8)
            else:
                if drawColor == (0, 0, 0):
                    cv2.line(image, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(image, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            cv2.circle(image,(x1,y1), 15, drawColor, cv2.FILLED)
            cv2.line(image,(xp,yp),(x1,y1),drawColor,brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp=x1,y1

            print(x1,y1)
    image[0:157, 0:1275] = header
    imgCanvas = cv2.resize(imgCanvas, (image.shape[1], image.shape[0]))

    image = cv2.addWeighted(image, 1, imgCanvas, 1, 1)

    cv2.imshow("Img", image)

    cv2.waitKey(1)
