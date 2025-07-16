import cv2
import numpy as np

# Load the age detection model
age_weights = "/home/sathwik-shetty/Desktop/CV/Age Detection/Dataset/age_net.caffemodel"
age_config = "/home/sathwik-shetty/Desktop/CV/Age Detection/Dataset/age_deploy.prototxt"
age_Net = cv2.dnn.readNet(age_config, age_weights)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        box = [x, y, x + w, y + h]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 200), 2)

        face = frame[box[1]:box[3], box[0]:box[2]]

        # Ensure the face region is large enough
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), model_mean, swapRB=False)

        age_Net.setInput(blob)
        age_preds = age_Net.forward()
        age = ageList[age_preds[0].argmax()]

        cv2.putText(frame, f'Age: {age}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Live Age Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
