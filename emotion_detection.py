import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = load_model("emotion_model.h5")
print("✅ Emotion model loaded")

# Emotion labels (same order as training)
emotions = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

# ===============================
# LOAD FACE DETECTOR
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# OPEN CAMERA
# ===============================
cap = cv2.VideoCapture(0)

print("📷 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # Crop face
        face = gray[y:y + h, x:x + w]

        try:
            # Resize to model input size
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            # Predict emotion
            preds = model.predict(face, verbose=0)
            emotion_label = emotions[np.argmax(preds)]

            # Display emotion
            cv2.putText(
                frame,
                emotion_label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        except:
            pass

    # Show output
    cv2.imshow("Emotion Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# RELEASE
# ===============================
cap.release()
cv2.destroyAllWindows()
