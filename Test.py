import cv2
import os

def Test():
    path = 'Datasets'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Trained/trainer.yml')
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Camera Setup
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)

    while True:
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if (confidence < 100):
                for imagePath in imagePaths:
                    if int(os.path.split(imagePath)[-1].split(".")[1]) == id:
                        name = str(os.path.split(imagePath)[-1].split(".")[0])
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                name = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.imshow('camera', img)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break
    print("\nExiting Program")
    camera.release()
    cv2.destroyAllWindows()

