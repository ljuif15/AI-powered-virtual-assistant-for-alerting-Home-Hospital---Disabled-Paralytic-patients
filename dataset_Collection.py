import cv2
import mediapipe as mp
import os
import csv

DATASET_DIR = "dataset"        # a b c d e f folders
OUTPUT_CSV = "hand_landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    # CSV Header
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)

    # Loop through folders
    for label in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_path):
            continue

        print(f"📁 Processing class: {label}")

        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Skip images where hand is not detected
            if not result.multi_hand_landmarks:
                continue

            landmarks = result.multi_hand_landmarks[0]
            row = []

            for lm in landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row.append(label)
            writer.writerow(row)

print("\n✅ hand_landmarks.csv CREATED SUCCESSFULLY")
