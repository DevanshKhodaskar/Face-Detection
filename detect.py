import cv2
import cv2.data
import numpy as np

video_path = "sampleVideo.mp4"
# Capture video from the file
cam = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cam.isOpened():
    print("Error: Could not open video.")
    exit()

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cam.read()

    # If the frame is read correctly, ret is True
    if not ret:
        print("Reached the end of the video or there is an error.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around faces and detect eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 9)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the frame with detected faces and eyes
    cv2.imshow("frame", frame)

    # Exit when 'x' is pressed
    if cv2.waitKey(34) == ord('x'):
        break

# Release video capture object and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
