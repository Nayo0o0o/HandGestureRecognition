import cv2
def main():
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("摄像头启动成功，按 Q 键退出")

    while True:
        # 读取摄像头画面
        ret, frame = cap.read()

        if not ret:
            print("无法获取摄像头画面")
            break

        # 显示画面
        cv2.imshow("Hand Gesture Recognition", frame)

        # 按 Q 键退出
        if cv2.waitKey(1) & 0xFF == ord('Q'):
            break

    # 释放摄像头
    cap.release()

    # 关闭所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
