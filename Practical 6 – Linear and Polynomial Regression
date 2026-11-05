import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

data = pd.read_csv("USA_Housing.csv")

print("First 5 records:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

X = data[
    [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Avg. Area Number of Bedrooms",
        "Area Population"
    ]
]

y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_mae = mean_absolute_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

print("\nLinear Regression:")
print("MSE =", linear_mse)
print("MAE =", linear_mae)
print("R2 =", linear_r2)

degree = 2

polynomial_model = make_pipeline(
    PolynomialFeatures(degree=degree),
    LinearRegression()
)

polynomial_model.fit(X_train, y_train)

y_pred_poly = polynomial_model.predict(X_test)

poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_mae = mean_absolute_error(y_test, y_pred_poly)
poly_r2 = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression:")
print("Degree =", degree)
print("MSE =", poly_mse)
print("MAE =", poly_mae)
print("R2 Score =", poly_r2)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression"
    ],
    "MSE": [linear_mse, poly_mse],
    "MAE": [linear_mae, poly_mae],
    "R2 Score": [linear_r2, poly_r2]
})

print("\nRegression Results:")
print(results)

for name, prediction in [
    ("Linear Regression", y_pred_linear),
    ("Polynomial Regression", y_pred_poly)
]:

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, prediction)

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(name + ": Actual vs Predicted Price")

    plt.show()

models = [
    "Linear Regression",
    "Polynomial Regression"
]

plt.figure(figsize=(7, 5))

plt.bar(models, [linear_r2, poly_r2])

plt.xlabel("Regression Model")
plt.ylabel("R2 Score")
plt.title("Comparison of R2 Scores")

plt.show()

plt.figure(figsize=(7, 5))

plt.bar(models, [linear_mse, poly_mse])

plt.xlabel("Regression Model")
plt.ylabel("Mean Squared Error")
plt.title("Comparison of MSE")

plt.show()

plt.figure(figsize=(7, 5))

plt.bar(models, [linear_mae, poly_mae])

plt.xlabel("Regression Model")
plt.ylabel("Mean Absolute Error")
plt.title("Comparison of MAE")

plt.show()
