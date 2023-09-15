import cv2

izuku = cv2.imread("faces/1.jpg")

gray = cv2.cvtColor(izuku, cv2.COLOR_BGR2GRAY)

haar_cascade = cv2.CascadeClassifier("haar_face.xml")

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f"number of faecs = {len(faces_rect)}")

for x, y, w, h in faces_rect:
    cv2.rectangle(izuku, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)

cv2.imshow("img", izuku)

cv2.waitKey(0)
