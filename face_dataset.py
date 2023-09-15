import cv2
import os

webcam = cv2.VideoCapture(0)
n = 1
while True:
    check, frame = webcam.read()
    # print(check)
    # print(frame)
    cv2.imshow("cam", frame)
    key = cv2.waitKey(1)

    if key == ord("s"):
        filename = os.path.join("faces/not_me", f"{n}.jpg")
        cv2.imwrite(filename, img=frame)
        n = n + 1

    if key == ord("q"):
        webcam.release()
        cv2.destroyAllWindows()
        break
