import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -----------------------------
# Load Titanic dataset
# -----------------------------

titanic = sns.load_dataset("titanic")

cleaned_titanic = titanic.copy()

cleaned_titanic = cleaned_titanic.drop(columns=["deck"])

cleaned_titanic = cleaned_titanic.dropna(
    subset=["embarked", "embark_town"]
)

cleaned_titanic["age"] = cleaned_titanic["age"].fillna(
    cleaned_titanic["age"].median()
)


# -----------------------------
# Features and target
# -----------------------------

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X = cleaned_titanic[features]

y = cleaned_titanic["fare"]


# -----------------------------
# Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------
# Feature groups
# -----------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]


# -----------------------------
# Preprocessing
# -----------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# -----------------------------
# Regression model
# -----------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    )
])


# -----------------------------
# Train
# -----------------------------

model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(X_test)


# -----------------------------
# Regression metrics
# -----------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# -----------------------------
# Adjusted R-squared
# -----------------------------

n = len(y_test)

p = X_test.shape[1]

adjusted_r2 = (
    1
    - ((1 - r2) * (n - 1) / (n - p - 1))
)


# -----------------------------
# Residuals
# -----------------------------

residuals = y_test - predictions


print("\n--- Fare Regression Results ---")

print("MAE:", mae)

print("RMSE:", rmse)

print("R-squared:", r2)

print("Adjusted R-squared:", adjusted_r2)


# -----------------------------
# Residual plot
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")

plt.ylabel("Residual")

plt.title("Residuals vs Predicted Fare")

plt.tight_layout()

plt.show()