import cv2
from deepface import DeepFace
from ultralytics import YOLO

# Load YOLO face model
model = YOLO("yolov8n-face.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (900, 900))
    copy = frame.copy()

    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face_img = frame[y1:y2, x1:x2]

            try:
                prediction = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False
                )

                dominant_emotion = prediction[0]['dominant_emotion']

                cv2.putText(
                    copy,
                    dominant_emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            except Exception as e:
                print("DeepFace error:", e)

    cv2.imshow("Face Detection", copy)

    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

cap.release()
cv2.destroyAllWindows()
