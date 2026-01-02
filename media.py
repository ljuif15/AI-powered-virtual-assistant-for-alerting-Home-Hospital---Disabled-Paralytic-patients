import cv2
import mediapipe as mp
import os
import numpy as np

INPUT_DIR = "dataset"
OUTPUT_DIR = "skeleton_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

for cls in os.listdir(INPUT_DIR):
    cls_path = os.path.join(INPUT_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    out_cls = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_cls, exist_ok=True)

    for img_name in os.listdir(cls_path):
        if not img_name.lower().endswith(('.jpg','.png','.jpeg')):
            continue

        img_path = os.path.join(cls_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ❌ Skip bad detections
        if not results.multi_hand_landmarks:
            continue

        # White background but scaled correctly
        skeleton = np.zeros_like(image)
        skeleton[:] = (255, 255, 255)

        for hand in results.multi_hand_landmarks:
            points = []
            for lm in hand.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))

            # Draw bones
            for s, e in HAND_CONNECTIONS:
                cv2.line(skeleton, points[s], points[e], (0,0,0), 2)

            # Draw joints
            for x, y in points:
                cv2.circle(skeleton, (x, y), 3, (0,0,0), -1)

        name, ext = os.path.splitext(img_name)
        cv2.imwrite(
            os.path.join(out_cls, f"{name}_skeleton{ext}"),
            skeleton
        )

print("✅ Proper hand skeletons generated")
