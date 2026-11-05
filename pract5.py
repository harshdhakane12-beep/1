import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv")

print("First Five Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nDuplicate Records:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

df.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1,
    inplace=True,
    errors="ignore"
)

print("\nDataset after Data Cleaning:")
print(df.head())

print("\nUpdated Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Titanic Dataset")
plt.show()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMean:")
print(df.mean(numeric_only=True))

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()

numerical_cols = ["Age", "SibSp", "Parch", "Fare"]

df[numerical_cols].hist(
    figsize=(12, 8),
    bins=20,
    edgecolor="black"
)

plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_cols])
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival Distribution by Gender")
plt.show()

X = df.drop("Survived", axis=1)
y = df["Survived"]

print("\nFeatures:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

numerical_cols = ["Age", "SibSp", "Parch", "Fare"]
categorical_cols = ["Pclass", "Sex", "Embarked"]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(
    "\nTraining Data after Preprocessing:",
    X_train_processed.shape
)

print(
    "Testing Data after Preprocessing:",
    X_test_processed.shape
)

feature_names = preprocessor.get_feature_names_out()

X_train_final = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

print("\nProcessed Training Dataset:")
print(X_train_final.head())

print("\n========== FINAL SUMMARY ==========")

print("Original Dataset Shape:", df.shape)
print("Training Dataset Shape:", X_train_final.shape)
print("Testing Dataset Shape:", X_test_processed.shape)

print(
    "\nMissing Values in Training Data:",
    X_train_final.isnull().sum().sum()
)

print(
    "Missing Values in Testing Data:",
    np.isnan(X_test_processed).sum()
)

print(
    "\nData Preprocessing and Exploratory "
    "Data Analysis Completed Successfully!"
)
