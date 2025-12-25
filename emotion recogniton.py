import cv2
import dlib
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# Initialize Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from Dlib

# Function to extract facial landmarks
def extract_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks_list = []
    
    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in shape.parts()]
        landmarks_list.append(landmarks)
    
    return landmarks_list

# Function to calculate geometric features
def calculate_features(landmarks):
    features = []
    for i, point in enumerate(landmarks):
        for j, point2 in enumerate(landmarks):
            if i < j:  # Calculate pairwise distances
                distance = np.linalg.norm(np.array(point) - np.array(point2))
                features.append(distance)
    return features
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        cv2.putText(frame,  (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()