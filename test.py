import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = tf.keras.models.load_model("hand_landmark_model.h5")

# ⚠️ MUST MATCH TRAINING LABEL ORDER
CLASSES = ['a', 'b', 'c', 'd', 'e', 'f']

# ===============================
# MEDIAPIPE HANDS
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# SMOOTHING (REDUCE FLICKER)
# ===============================
prediction_queue = deque(maxlen=10)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

print("📷 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw hand landmarks (optional – for visualization)
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # ===============================
        # EXTRACT 21 LANDMARKS
        # ===============================
        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])

        data = np.array(data).reshape(1, -1)

        # ===============================
        # PREDICT
        # ===============================
        probs = model.predict(data, verbose=0)[0]
        class_id = np.argmax(probs)
        confidence = probs[class_id]

        prediction_queue.append(class_id)
        final_class = max(set(prediction_queue), key=prediction_queue.count)

        label = CLASSES[final_class]

        # ===============================
        # DISPLAY RESULT
        # ===============================
        cv2.rectangle(frame, (10,10), (320,90), (0,0,0), -1)
        cv2.putText(frame, f"Gesture: {label}", (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Real-Time Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
