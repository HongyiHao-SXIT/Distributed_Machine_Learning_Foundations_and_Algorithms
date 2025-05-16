import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score

# 1. 加载数据
df = pd.read_csv("train.csv")

# 2. 将 Age 缺失值填充为 0
df['Age'] = df['Age'].fillna(0)

# 3. 其他字段缺失值统一填充为 0
df.fillna(0, inplace=True)

# 4. OneHot 编码类别变量
categorical_cols = df.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# 5. 特征与标签划分
X = df_encoded.drop("Survived", axis=1)
y = df_encoded["Survived"]

# 6. 拆分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 处理不平衡标签
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# 8. 使用 KNN（k=5）训练模型并预测
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_resampled, y_train_resampled)
y_pred = knn.predict(X_test)

# 9. 输出测试准确率
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (Age filled with 0): {test_accuracy:.4f}")
