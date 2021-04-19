import cv2
import numpy as np
from urllib.request import urlopen
import os
import datetime
import time
import sys

import cv2
import mediapipe as mp

def hand_detection(image_name, index):
#   print(dir(mp))
  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands

  # For static images:
  hands = mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5)
  # Read an image, flip it around y-axis for correct handedness output (see
  # above).
  idx = 0
  image = cv2.flip(image_name, 1)
  # Convert the BGR image to RGB before processing.
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Print handedness and draw hand landmarks on the image.
  print('Handedness:', results.multi_handedness)
  if not results.multi_hand_landmarks:
    print("WARNING: This image has no hand(s)!!!")
    return image_name
  else:
    index = index + 1
    image_hight, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
      )
      mp_drawing.draw_landmarks(
          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    print("hand landmarks drawing done!")
    # relative_path = '/tmp/annotated_image/' + str(idx) + '.png'
    # cv2.imwrite(relative_path, cv2.flip(annotated_image, 1))
    output = cv2.flip(annotated_image, 1)
    cv2.imwrite("output_" + str(index) + ".jpg", output)
  hands.close()

  return output

url="http://192.168.43.42"
CAMERA_BUFFER_SIZE=4096
stream=urlopen(url + "/stream.jpg")
bts=b''
i=0

idx = 0 # for output image

while True:    
    try:
        idx_prev = idx
        bts+=stream.read(CAMERA_BUFFER_SIZE)
        jpghead=bts.find(b'\xff\xd8')
        jpgend=bts.find(b'\xff\xd9')
        if jpghead>-1 and jpgend>-1:
            jpg=bts[jpghead:jpgend+2]
            bts=bts[jpgend+2:]
            img=cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            img=cv2.resize(img,(640,480))
            proc_img = hand_detection(img,idx)
            cv2.imshow("ESP32 CAM OPENCV stream",proc_img)
            if idx > idx_prev:
                time.sleep(3)
                print("image saved, sleep for 3s")
        k=cv2.waitKey(1)
    except Exception as e:
        print("Error:" + str(e))
        bts=b''
        stream=urlopen(url)
        continue
    # Press 'a' to take a picture
    if k & 0xFF == ord('a'):
        cv2.imwrite(str(i) + ".jpg", img)
        print(f"Save image filename: {i}.jpg")
        i=i+1
    # Press 'q' to quit
    if k & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
