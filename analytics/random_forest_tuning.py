import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# -----------------------------
# Load and clean data
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
    "fare",
    "embarked"
]

X = cleaned_titanic[features]
y = cleaned_titanic["survived"]


# -----------------------------
# Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# Preprocessing
# -----------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

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
# Random Forest pipeline
# -----------------------------

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            oob_score=True,
            bootstrap=True
        )
    )
])


# -----------------------------
# GridSearchCV
# -----------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


grid_search.fit(X_train, y_train)


# -----------------------------
# Best model
# -----------------------------

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]


print("\n--- Random Forest Grid Search ---")

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1:")
print(grid_search.best_score_)

print("\nOOB score:")
print(best_model.named_steps["model"].oob_score_)


print("\n--- Test Set Metrics ---")

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))