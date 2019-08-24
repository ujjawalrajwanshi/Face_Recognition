import cv2
import numpy as np
from PIL import Image
import os

path = 'Datasets'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def getImages(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\n Training faces.Wait ...")
face,ids = getImages(path)
recognizer.train(face, np.array(ids))


recognizer.write('Trained/trainer.yml')


print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))
