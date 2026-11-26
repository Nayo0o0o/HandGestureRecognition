
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. 读取数据
right_data = pd.read_csv("data/hand_gestures.csv")
left_data = pd.read_csv("data/left_hand_gestures.csv")

data = pd.concat(
    [right_data, left_data],
    ignore_index=True
)

X = data.drop("label", axis=1)
y = data["label"]

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. 训练模型
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# 4. 预测
y_pred = model.predict(X_test)

print("准确率：", accuracy_score(y_test, y_pred))

print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 5. 混淆矩阵
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Hand Gesture Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 6. 各类别指标对比
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

metrics = pd.DataFrame(report).T
metrics = metrics.loc[
    model.classes_,
    ["precision", "recall", "f1-score"]
]

metrics.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Classification Performance")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("classification_metrics.png", dpi=300)
plt.show()

print("\n评估图表已保存！")
print("confusion_matrix.png")
print("classification_metrics.png")
