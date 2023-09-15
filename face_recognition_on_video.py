import numpy as np
import cv2

haar_cascade = cv2.CascadeClassifier("haar_face.xml")

webcam = cv2.VideoCapture(0)

people = ["me", "not_me"]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# img = cv.imread(r"faces/test.jpg")


while True:
    check, frame = webcam.read()

    cv2.imshow("cam", frame)
    key = cv2.waitKey(100)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Person", gray)

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces_rect:
        faces_roi = gray[y : y + h, x : x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f"Label = {people[label]} with a confidence of {confidence}")
        if confidence > 20:
            cv2.putText(
                frame,
                str(people[label] + " " + str(confidence)),
                (20, 20),
                cv2.FONT_HERSHEY_COMPLEX,
                1.0,
                (0, 255, 0),
                thickness=2,
            )
        else:
            cv2.putText(
                frame,
                str("none" + " " + str(confidence)),
                (20, 20),
                cv2.FONT_HERSHEY_COMPLEX,
                1.0,
                (0, 255, 0),
                thickness=2,
            )

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        cv2.imshow("Detected Face", frame)

    if key == ord("q"):
        webcam.release()
        cv2.destroyAllWindows()
        break
