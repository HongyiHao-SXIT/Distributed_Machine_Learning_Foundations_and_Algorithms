import warnings
import os

# 屏蔽未来弃用警告和用户警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 避免loky并行库报错，限制使用1个CPU核
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score

# 1. Read and Visualize Data
def load_data():
    df = pd.read_csv("train.csv")
    print("First 5 rows of data:")
    print(df.head())
    return df

# 2. Handling Missing Data
def fill_missing(df):
    print("\nMissing values before:")
    print(df.isnull().sum())
    df_filled = df.fillna(0)
    print("\nMissing values after:")
    print(df_filled.isnull().sum())
    return df_filled

# 3. Handling numerical and categorical values
def identify_columns(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\nNumerical columns:", numerical_cols)
    print("Categorical columns:", categorical_cols)
    return numerical_cols, categorical_cols

def encode_categorical(df, categorical_cols):
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    print("\nFirst 5 rows after encoding:")
    print(df_encoded.head())
    return df_encoded

# 4. Dataset Splitting
def split_dataset(df):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nShapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    return X_train, X_test, y_train, y_test

# 5. Addressing Imbalanced Labels
def handle_imbalance(X_train, y_train):
    print("\nClass distribution before:")
    print(y_train.value_counts())
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    print("Class distribution after:")
    print(y_resampled.value_counts())
    return X_resampled, y_resampled

# 6. Train KNN and Evaluate
def train_knn(X_train, y_train, X_test, y_test):
    accuracies_train = []
    accuracies_test = []
    k_range = range(1, 101)
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        accuracies_train.append(accuracy_score(y_train, y_pred_train))
        accuracies_test.append(accuracy_score(y_test, y_pred_test))

    # Plot accuracy vs. k
    plt.figure(figsize=(10,6))
    plt.plot(k_range, accuracies_train, label='Train Accuracy')
    plt.plot(k_range, accuracies_test, label='Test Accuracy')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs k")
    plt.legend()
    plt.grid()
    plt.savefig("screenshots/knn_accuracy.png")
    plt.show()

    print(f"\nBest test accuracy: {max(accuracies_test):.4f} at k = {accuracies_test.index(max(accuracies_test))+1}")

# Main function to run all steps
def main():
    df = load_data()
    df_filled = fill_missing(df)
    numerical_cols, categorical_cols = identify_columns(df_filled)
    df_encoded = encode_categorical(df_filled, categorical_cols)
    X_train, X_test, y_train, y_test = split_dataset(df_encoded)
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train)
    train_knn(X_train_bal, y_train_bal, X_test, y_test)

if __name__ == "__main__":
    main()
