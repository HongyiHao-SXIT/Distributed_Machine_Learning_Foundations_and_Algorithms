import pandas as pd

df = pd.read_csv("train.csv")
df.fillna(0, inplace=True)

# (a) Identify numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print("Numerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

# (b) Convert categorical columns into one-hot encoded columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("\nFirst 5 rows after one-hot encoding:")
print(df_encoded.head())
