import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

data = pd.read_csv("breast-cancer.csv")

print("First 5 records:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

data = data.drop("Survival Months", axis=1)

X = data.drop("Status", axis=1)

y = data["Status"].map({
    "Alive": 1,
    "Dead": 0
})

numerical_columns = X.select_dtypes(
    include=["number"]
).columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

print("\nNumerical Columns:", list(numerical_columns))
print("Categorical Columns:", list(categorical_columns))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

logistic_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

precision_lr = precision_score(
    y_test, y_pred_lr, zero_division=0
)

recall_lr = recall_score(
    y_test, y_pred_lr, zero_division=0
)

f1_lr = f1_score(
    y_test, y_pred_lr, zero_division=0
)

cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\nLOGISTIC REGRESSION RESULTS")

print("Accuracy :", accuracy_lr)
print("Precision:", precision_lr)
print("Recall   :", recall_lr)
print("F1-Score :", f1_lr)

print("\nConfusion Matrix:")
print(cm_lr)

knn_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)

precision_knn = precision_score(
    y_test, y_pred_knn, zero_division=0
)

recall_knn = recall_score(
    y_test, y_pred_knn, zero_division=0
)

f1_knn = f1_score(
    y_test, y_pred_knn, zero_division=0
)

cm_knn = confusion_matrix(y_test, y_pred_knn)

print("\nKNN RESULTS")

print("Accuracy :", accuracy_knn)
print("Precision:", precision_knn)
print("Recall   :", recall_knn)
print("F1-Score :", f1_knn)

print("\nConfusion Matrix:")
print(cm_knn)

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],

    "Logistic Regression": [
        accuracy_lr,
        precision_lr,
        recall_lr,
        f1_lr
    ],

    "KNN": [
        accuracy_knn,
        precision_knn,
        recall_knn,
        f1_knn
    ]
})

print("\nMODEL COMPARISON")
print(comparison)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Dead", "Alive"]
).plot(ax=axes[0])

axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay(
    confusion_matrix=cm_knn,
    display_labels=["Dead", "Alive"]
).plot(ax=axes[1])

axes[1].set_title("KNN")

plt.tight_layout()
plt.show()
