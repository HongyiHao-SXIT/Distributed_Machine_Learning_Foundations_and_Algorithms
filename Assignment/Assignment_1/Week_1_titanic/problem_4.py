# 导入所需库
from sklearn.model_selection import train_test_split
import pandas as pd

# 假设 df_encoded 是处理好、并完成 One-Hot 编码后的 DataFrame
# 并且已经包含目标列 'Survived'

# 特征和标签划分
X = df_encoded.drop(columns=['Survived'])  # 特征部分
y = df_encoded['Survived']                 # 标签（目标值）

# 划分为训练集和测试集，20% 用作测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 输出各数据集的形状
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
