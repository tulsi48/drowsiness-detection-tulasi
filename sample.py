import cv2, dlib, imutils
from imutils import face_utils
from scipy.spatial import distance as dist

print("Starting camera...")
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Camera not detected! Exiting.")
    exit()

print("Loading face detector and predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("System ready ✅")
count = 0
earThresh = 0.3
earFrames = 48

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Frame not captured.")
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        A = dist.euclidean(leftEye[1], leftEye[5])
        B = dist.euclidean(leftEye[2], leftEye[4])
        C = dist.euclidean(leftEye[0], leftEye[3])
        ear = (A + B) / (2.0 * C)

        if ear < earThresh:
            count += 1
            if count >= earFrames:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            count = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
