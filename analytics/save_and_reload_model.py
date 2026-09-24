import pandas as pd
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Load Titanic data
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
# Complete pipeline
# -----------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# -----------------------------
# Fit
# -----------------------------

model_pipeline.fit(X_train, y_train)


# -----------------------------
# Save
# -----------------------------

model_path = "analytics/titanic_survival_pipeline.joblib"

joblib.dump(
    model_pipeline,
    model_path
)

print("Model saved:", model_path)


# -----------------------------
# Reload
# -----------------------------

loaded_model = joblib.load(model_path)

print("Model reloaded successfully!")


# -----------------------------
# Raw input prediction
# -----------------------------

raw_input = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 100,
    "embarked": "S"
}])


prediction = loaded_model.predict(raw_input)

probability = loaded_model.predict_proba(raw_input)[0][1]


print("\n--- Raw Input ---")
print(raw_input)

print("\nPrediction:", prediction[0])

print("Survival probability:", probability)