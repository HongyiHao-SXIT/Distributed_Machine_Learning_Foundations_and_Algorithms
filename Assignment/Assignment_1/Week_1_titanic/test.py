import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import sklearn

import pandas as pd

# 读取数据
df = pd.read_csv("train.csv")

# （a）识别数值型和类别型列
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("=== Numerical Columns ===")
print(numerical_cols)

print("\n=== Categorical Columns ===")
print(categorical_cols)

# （b）对类别型特征进行 One-Hot 编码
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# 打印编码后的前 5 行
print("\n=== First 5 rows after One-Hot Encoding ===")
print(df_encoded.head())
