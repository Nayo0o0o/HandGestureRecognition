
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# 1. 读取左右手数据
# =========================

right_path = "data/hand_gestures.csv"
left_path = "data/left_hand_gestures.csv"

if not os.path.exists(right_path) or not os.path.exists(left_path):
    raise FileNotFoundError("找不到左右手数据文件，请检查 data 文件夹")

right_data = pd.read_csv(right_path)
left_data = pd.read_csv(left_path)

print("右手数据量：", len(right_data))
print("左手数据量：", len(left_data))

# =========================
# 2. 合并数据
# =========================

data = pd.concat(
    [right_data, left_data],
    ignore_index=True
)

print("\n合并后数据总量：", len(data))
print("\n各类手势数量：")
print(data["label"].value_counts())

# 检查特征列
feature_columns = [f"feature_{i}" for i in range(63)]

if not all(col in data.columns for col in feature_columns):
    raise ValueError("数据集特征列不完整")

X = data[feature_columns]
y = data["label"]

# =========================
# 3. 划分训练集和测试集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. 创建并训练模型
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# 5. 模型评估
# =========================

y_pred = model.predict(X_test)

print("\n模型训练完成！")

print(
    f"\n测试集准确率："
    f"{accuracy_score(y_test, y_pred):.4f}"
)

print("\n分类报告：")
print(classification_report(y_test, y_pred))

print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# =========================
# 6. 保存模型
# =========================

joblib.dump(model, "hand_gesture_model.pkl")

print("\n模型已保存：hand_gesture_model.pkl")
