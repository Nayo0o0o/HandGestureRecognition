
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

# =========================
# 1. 加载模型
# =========================

model = joblib.load("hand_gesture_model.pkl")

FEATURE_COLUMNS = [f"feature_{i}" for i in range(63)]


# =========================
# 2. 特征提取
# =========================

def extract_features(hand_landmarks, handedness):

    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ])

    # 以手腕为原点
    points = points - points[0]

    # 左手进行 X 坐标翻转
    if handedness == "Left":
        points[:, 0] = -points[:, 0]

    # 尺度归一化
    scale = np.max(
        np.linalg.norm(points[:, :2], axis=1)
    )

    if scale == 0:
        return None

    points = points / scale

    return points.flatten()


# =========================
# 3. 主程序
# =========================

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print("双手实时手势识别已启动")
        print("支持左右手同时识别")
        print("按 Q 退出")

        while True:

            ret, frame = cap.read()

            if not ret:
                print("无法获取摄像头画面")
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            # 显示区域
            y_position = 40

            if results.multi_hand_landmarks:

                for hand, handedness_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    # 获取 MediaPipe 左右手标签
                    mp_label = (
                        handedness_info.classification[0].label
                    )

                    # 当前输入为未水平翻转的原始画面
                    handedness = (
                        "Right" if mp_label == "Left"
                        else "Left"
                    )

                    # 绘制关键点
                    mp_drawing.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # 提取特征
                    features = extract_features(
                        hand,
                        handedness
                    )

                    if features is None:
                        continue

                    X = pd.DataFrame(
                        [features],
                        columns=FEATURE_COLUMNS
                    )

                    # 模型预测
                    prediction = model.predict(X)[0]

                    probabilities = model.predict_proba(X)[0]

                    confidence = np.max(probabilities)

                    # 显示结果
                    result_text = (
                        f"{handedness}: "
                        f"{prediction} "
                        f"{confidence:.2%}"
                    )

                    cv2.putText(
                        frame,
                        result_text,
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    y_position += 40

            else:

                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Two-Hand Gesture Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
