import cv2
import time
import threading
import pygame

def highlightFace(faceCascade, frame):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceBoxes = faceCascade.detectMultiScale(frameGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    frameOpencvHaar = frame.copy()
    for (x, y, w, h) in faceBoxes:
        cv2.rectangle(frameOpencvHaar, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frameOpencvHaar, faceBoxes

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
pygame.mixer.init()
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderList = ['Male', 'Female']

genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(0)
padding = 20

last_played_time = 0
sound_interval = 10

def play_alert(arg):
    if arg=="Male":
        pygame.mixer.music.load("m.mp3")
        pygame.mixer.music.play()
    elif arg=="Female":
        pygame.mixer.music.load("f.mp3")
        pygame.mixer.music.play()

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceCascade, frame)
    if len(faceBoxes) == 0:
        cv2.imshow("Detecting gender", resultImg)

    for (x, y, w, h) in faceBoxes:
        face = frame[max(0, y - padding):min(y + h + padding, frame.shape[0] - 1),
                     max(0, x - padding):min(x + w + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting gender", resultImg)

        current_time = time.time()
        if current_time - last_played_time > sound_interval:
            threading.Thread(target=play_alert,args=(gender,)).start()
            last_played_time = current_time

video.release()
cv2.destroyAllWindows()
