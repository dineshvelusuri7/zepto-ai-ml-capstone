import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Titanic dataset from seaborn
titanic = sns.load_dataset("titanic")


# Save immediately as an offline fallback
titanic.to_csv("analytics/titanic.csv", index=False)


print("Titanic dataset loaded successfully!")
print("Shape:", titanic.shape)

print("\nColumns:")
print(titanic.columns.tolist())

print("\nFirst 5 rows:")
print(titanic.head())


print("\n--- Dataset Information ---")
titanic.info()

print("\n--- Statistical Summary ---")
print(titanic.describe())

print("\n--- Missing Values ---")
missing = titanic.isnull().sum()
missing_percent = (missing / len(titanic)) * 100

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
})

print(missing_summary)


print("\nExact missing percentages:")

for column in ["age", "embarked", "deck", "embark_town"]:
    percentage = titanic[column].isnull().mean() * 100
    print(f"{column}: {percentage:.2f}%")


    # Make a copy so we preserve the original loaded dataset
cleaned_titanic = titanic.copy()

# Drop the very-high-missingness column
cleaned_titanic = cleaned_titanic.drop(columns=["deck"])

# Drop rows where low-missingness columns are missing
cleaned_titanic = cleaned_titanic.dropna(
    subset=["embarked", "embark_town"]
)

# Impute age using the median
age_median = cleaned_titanic["age"].median()

cleaned_titanic["age"] = cleaned_titanic["age"].fillna(age_median)

print("\n--- After Cleaning ---")
print("Original shape:", titanic.shape)
print("Cleaned shape:", cleaned_titanic.shape)

print("\nRemaining missing values:")
print(cleaned_titanic.isnull().sum())


# -----------------------------
# Univariate Analysis
# -----------------------------

# Histogram for age
plt.figure(figsize=(8, 5))
plt.hist(cleaned_titanic["age"], bins=20)
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.title("Age Distribution")
plt.show()


# Box plot for age
plt.figure(figsize=(8, 5))
plt.boxplot(cleaned_titanic["age"])
plt.ylabel("Age")
plt.title("Age Box Plot")
plt.show()


# Histogram for fare
plt.figure(figsize=(8, 5))
plt.hist(cleaned_titanic["fare"], bins=20)
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.title("Fare Distribution")
plt.show()


# Box plot for fare
plt.figure(figsize=(8, 5))
plt.boxplot(cleaned_titanic["fare"])
plt.ylabel("Fare")
plt.title("Fare Box Plot")
plt.show()


def count_iqr_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return q1, q3, iqr, lower_bound, upper_bound, len(outliers)


age_results = count_iqr_outliers(cleaned_titanic, "age")
fare_results = count_iqr_outliers(cleaned_titanic, "fare")


print("\n--- IQR Outlier Analysis ---")

print("Age:")
print("Q1:", age_results[0])
print("Q3:", age_results[1])
print("IQR:", age_results[2])
print("Lower bound:", age_results[3])
print("Upper bound:", age_results[4])
print("Outlier count:", age_results[5])

print("\nFare:")
print("Q1:", fare_results[0])
print("Q3:", fare_results[1])
print("IQR:", fare_results[2])
print("Lower bound:", fare_results[3])
print("Upper bound:", fare_results[4])
print("Outlier count:", fare_results[5])



fare_mean = cleaned_titanic["fare"].mean()
fare_median = cleaned_titanic["fare"].median()
fare_mode = cleaned_titanic["fare"].mode()[0]

print("\n--- Fare Statistics ---")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)



# -----------------------------
# Bivariate Survival Analysis
# -----------------------------

print("\n--- Survival Rate by Sex ---")

survival_by_sex = (
    cleaned_titanic
    .groupby("sex")["survived"]
    .mean()
)

print(survival_by_sex)


print("\n--- Survival Rate by Passenger Class ---")

survival_by_class = (
    cleaned_titanic
    .groupby("pclass")["survived"]
    .mean()
)

print(survival_by_class)


print("\n--- Survival Rate by Sex and Passenger Class ---")

survival_by_sex_class = (
    cleaned_titanic
    .groupby(["sex", "pclass"])["survived"]
    .mean()
)

print(survival_by_sex_class)



# -----------------------------
# Correlation Analysis
# -----------------------------

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = cleaned_titanic[correlation_columns].corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix)


# Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Titanic Correlation Heatmap")
plt.tight_layout()
plt.show()


# Find the two strongest absolute correlations
upper_triangle = correlation_matrix.where(
    ~np.tril(np.ones(correlation_matrix.shape)).astype(bool)
)

correlation_pairs = (
    upper_triangle
    .stack()
    .abs()
    .sort_values(ascending=False)
)

print("\n--- Two Strongest Correlations ---")
print(correlation_pairs.head(2))


# -----------------------------
# Multivariate Visualizations
# -----------------------------

# Chart 1: Survival rate by sex and class
plt.figure(figsize=(8, 5))

sns.barplot(
    data=cleaned_titanic,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Passenger Class and Sex")
plt.tight_layout()
plt.show()


# Chart 2: Age distribution by survival
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_titanic,
    x="survived",
    y="age"
)

plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.show()


# Chart 3: Fare distribution by survival
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_titanic,
    x="survived",
    y="fare"
)

plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare")
plt.title("Fare Distribution by Survival")
plt.tight_layout()
plt.show()


# Chart 4: Survival rate by class and sex
plt.figure(figsize=(8, 5))

survival_table = (
    cleaned_titanic
    .groupby(["pclass", "sex"])["survived"]
    .mean()
    .reset_index()
)

sns.barplot(
    data=survival_table,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.title("Survival Rate Across Class and Sex")
plt.tight_layout()
plt.show()


# -----------------------------
# Standardization Check
# -----------------------------

scaler = StandardScaler()

standardized_values = scaler.fit_transform(
    cleaned_titanic[["age", "fare"]]
)

standardized_df = pd.DataFrame(
    standardized_values,
    columns=["age_standardized", "fare_standardized"]
)

print("\n--- Standardization Check ---")
print(standardized_df.head())

print("\nStandardized means:")
print(standardized_df.mean())

print("\nStandardized standard deviations:")
print(standardized_df.std())


# -----------------------------
# Train/Test Split
# -----------------------------

X = cleaned_titanic.drop(columns=["survived"])
y = cleaned_titanic["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- Train/Test Split ---")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))