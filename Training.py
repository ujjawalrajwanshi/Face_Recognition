import cv2


def Training():
    # Camera Setup
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # For each person,Assign an ID
    face_id = input('\nEnter user id and press\n')
    name = input("\nEnter user name\n ")
    print("\nCapturing Face wait\n")
    count = 0
    #loop for capturing images
    while (True):
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite('Datasets/' + name + '.' + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)
        key = cv2.waitKey(100) & 0xff  # Press 'Esc' to Exit
        if key == 27:
            break
        elif count >= 27:
            break

    print("\nSample collection Successful !\n")
    camera.release()
    cv2.destroyAllWindows()

