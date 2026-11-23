
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from collections import Counter

GESTURES = {
    ord('1'): "fist",
    ord('2'): "palm",
    ord('3'): "peace",
    ord('4'): "thumbs_up",
    ord('5'): "ok"
}

DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "hand_gestures.csv")

TARGET = 200
INTERVAL = 0.2


def extract_features(hand_landmarks):
    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ])

    points = points - points[0]

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))

    if scale == 0:
        return None

    points = points / scale

    return points.flatten()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    counts = Counter()

    # 读取已有数据，避免重复采集
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                counts[row["label"]] += 1

    file_exists = os.path.exists(CSV_FILE)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writerow(
                ["label"] + [f"feature_{i}" for i in range(63)]
            )

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            selected_label = None
            paused = False
            last_save_time = 0

            print("自动数据采集已启动")
            print("按 1-5 选择手势，空格暂停，Q退出")

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                features = None

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    mp_drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    features = extract_features(hand)

                # 自动采集
                current_time = time.time()

                if (
                    selected_label is not None
                    and not paused
                    and features is not None
                    and current_time - last_save_time >= INTERVAL
                ):
                    if counts[selected_label] < TARGET:
                        writer.writerow(
                            [selected_label] + features.tolist()
                        )
                        file.flush()

                        counts[selected_label] += 1
                        last_save_time = current_time

                        print(
                            f"{selected_label}: "
                            f"{counts[selected_label]}/{TARGET}"
                        )

                    if counts[selected_label] >= TARGET:
                        print(f"{selected_label} 采集完成！")
                        selected_label = None

                # 显示操作说明
                cv2.putText(
                    frame,
                    "1:Fist 2:Palm 3:Peace 4:ThumbsUp 5:OK",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                if selected_label is not None:
                    status = (
                        f"Collecting: {selected_label} "
                        f"{counts[selected_label]}/{TARGET}"
                    )
                    if paused:
                        status = "PAUSED - " + status
                else:
                    status = "Select gesture: Press 1-5"

                cv2.putText(
                    frame,
                    status,
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.imshow("Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key in GESTURES:
                    selected_label = GESTURES[key]
                    paused = False

                    if counts[selected_label] >= TARGET:
                        print(f"{selected_label} 已达到 {TARGET} 条")
                        selected_label = None
                    else:
                        print(f"开始采集：{selected_label}")

                elif key == ord(' '):
                    if selected_label is not None:
                        paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    print("\n数据采集结束")
    print("各类样本数量：")
    for label in GESTURES.values():
        print(f"{label}: {counts[label]}/{TARGET}")


if __name__ == "__main__":
    main()
