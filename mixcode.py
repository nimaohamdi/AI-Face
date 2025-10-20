import cv2
import numpy as np

# Initialize camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Cannot access webcam")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Random colors for visualization
color = np.random.randint(0, 255, (100, 3))

# Initialize variables
old_gray = None
p0 = None
mask = None

print("✅ Press [ESC] to exit")

while True:
    ret, frame = cam.read()
    if not ret:
        print("⚠️ Frame not received. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest for features
        roi_gray = gray[y:y + h, x:x + w]

        # Initialize tracking points if not available
        if p0 is None or old_gray is None:
            p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
            if p0 is not None:
                p0[:, 0, 0] += x
                p0[:, 0, 1] += y
                old_gray = gray.copy()
                mask = np.zeros_like(frame)
            continue

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw tracking lines
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            # Overlay the mask on the frame
            frame = cv2.add(frame, mask)

            # Update previous frame and points
            old_gray = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    cv2.imshow('Face Detection + Optical Flow Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
