import cv2
import numpy as np

cam = cv2.VideoCapture(0)  

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


params = dict(winSize=(20, 20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

       
        p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **params)
        if p0 is not None:
            p0 = np.float32(p0).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, gray_frame, p0, None, **lk_params)
            if st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(frame, (a + x, b + y), (c + x, d + y), (0, 255, 0), 2)
                    cv2.circle(frame, (a + x, b + y), 5, (0, 255, 0), -1)

    cv2.imshow('Webcam Face Detection and Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break


cam.release()
cv2.destroyAllWindows()
