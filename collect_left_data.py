
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from collections import Counter

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "left_hand_gestures.csv")

TARGET_COUNT = 200
SAVE_INTERVAL = 0.2

LABELS = {
    "1": "fist",
    "2": "palm",
    "3": "peace",
    "4": "thumbs_up",
    "5": "ok"
}

FEATURE_COLUMNS = [f"feature_{i}" for i in range(63)]
CSV_COLUMNS = ["label"] + FEATURE_COLUMNS


def extract_features(hand_landmarks):
    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ])

    # 以手腕为原点
    points = points - points[0]

    # 将左手转换到统一坐标方向
    points[:, 0] = -points[:, 0]

    # 尺度归一化
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))

    if scale == 0:
        return None

    points = points / scale

    return points.flatten()


def load_counts():
    counts = Counter()

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                counts[row["label"]] += 1

    return counts


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    file_exists = os.path.exists(CSV_PATH)

    if not file_exists:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

    counts = load_counts()

    print("左手数据采集程序")
    print("每种手势目标数量：", TARGET_COUNT)
    print("按 1-5 选择手势，按 Q 退出")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    selected_label = None
    last_save_time = 0

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("无法获取摄像头画面")
                    break

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb_frame)

                hand_text = "No hand"

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    # 原始画面没有水平翻转，因此交换 MediaPipe 标签
                    mp_label = (
                        results.multi_handedness[0]
                        .classification[0].label
                    )

                    actual_hand = (
                        "Right" if mp_label == "Left" else "Left"
                    )

                    hand_text = actual_hand

                    mp_drawing.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

                    if actual_hand == "Left" and selected_label:
                        if counts[selected_label] < TARGET_COUNT:
                            current_time = time.time()

                            if current_time - last_save_time >= SAVE_INTERVAL:
                                features = extract_features(hand)

                                if features is not None:
                                    with open(
                                        CSV_PATH,
                                        "a",
                                        newline="",
                                        encoding="utf-8"
                                    ) as f:
                                        writer = csv.writer(f)
                                        writer.writerow(
                                            [selected_label] + features.tolist()
                                        )

                                    counts[selected_label] += 1
                                    last_save_time = current_time

                cv2.putText(
                    frame,
                    f"Hand: {hand_text}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                if selected_label:
                    count = counts[selected_label]

                    cv2.putText(
                        frame,
                        f"Gesture: {selected_label}",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Samples: {count}/{TARGET_COUNT}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                cv2.putText(
                    frame,
                    "1:Fist 2:Palm 3:Peace 4:ThumbsUp 5:OK",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                cv2.imshow("Left Hand Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if chr(key) in LABELS:
                    selected_label = LABELS[chr(key)]

                    if counts[selected_label] >= TARGET_COUNT:
                        print(selected_label, "已经采集完成")
                    else:
                        print("开始采集：", selected_label)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n左手数据采集统计：")
    for label in LABELS.values():
        print(f"{label}: {counts[label]}/{TARGET_COUNT}")


if __name__ == "__main__":
    main()
