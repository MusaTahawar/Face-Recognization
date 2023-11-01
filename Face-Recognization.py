import cv2
import dlib

# Load face detection and shape predictor models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Create a face recognition model (in this example, we'll use LBPH)
recognizer = cv2.face_LBPHFaceRecognizer.create()

# Train the face recognizer
recognizer.train(faces_data, labels)

# Initialize the webcam (or load an image)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or provide a filename for an image

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face features
        shape = predictor(gray, dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h))
        # Extract specific facial landmarks if needed

        # Recognize the face
        label, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the recognized label and confidence
        cv2.putText(frame, f"Label: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
