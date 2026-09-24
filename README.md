## Module 1 - Data Pipeline

### Data Source
Books to Scrape: https://books.toscrape.com/

The pipeline scrapes books from three categories:
- Mystery
- Historical Fiction
- Fiction

### Data Processing
The scraper collects:
- title
- price_gbp
- rating
- in_stock
- category

Price is converted from GBP to INR using the fixed rate:

1 GBP = 105.50 INR

The final dataset contains 123 records with no missing values or duplicate titles.

### Database
The cleaned dataset is stored in SQLite using two related tables:
- categories
- books

The `category_id` field connects the two tables through a foreign key.

### SQL Analysis
The project includes SQL queries demonstrating:
- SELECT and WHERE
- ORDER BY and LIMIT
- DISTINCT
- IN
- BETWEEN
- JOIN

SQL results are saved in:

`data_pipeline/sql_outputs.txt`

The JOIN result is also reproduced using `pandas.merge()`.

## Module 2 — Titanic Analytics and Machine Learning

### Dataset

The Titanic dataset is loaded from Seaborn and saved locally as `analytics/titanic.csv` so the analysis can be reproduced offline.

### Data Cleaning

Missing-value handling follows the required threshold-based strategy:

* `age` — missing values are imputed because the missing percentage is between 5% and 30%.
* `embarked` and `embark_town` — rows with missing values are removed because their missing percentage is below 5%.
* `deck` — the column is dropped because more than 30% of its values are missing.

### Exploratory Data Analysis

The analysis includes:

* Dataset shape, information, descriptive statistics and missing-value percentages.
* Age and fare distributions using histograms and boxplots.
* IQR-based outlier identification for age and fare.
* Fare mean, median, mode and skewness.
* Survival rates by sex, passenger class, and sex combined with passenger class.
* Correlation matrix and heatmap using `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`.
* Multivariate visualizations with written interpretations.

The strongest absolute correlations are:

* `pclass` and `fare`: approximately `-0.548`
* `sibsp` and `parch`: approximately `0.415`

### Machine Learning Classification

The classification workflow uses a stratified train/test split followed by preprocessing fitted only on the training data.

Models evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

The preprocessing uses `Pipeline` and `ColumnTransformer`. Classification performance is evaluated using accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC.

### Class Imbalance

Three approaches are compared:

* Baseline model
* Class-weighted model
* SMOTE applied only to the training data

### Random Forest Tuning

`GridSearchCV` is used to tune:

* `n_estimators`
* `max_depth`
* `max_features`

The tuned Random Forest uses `oob_score=True`, and the best parameters and OOB score are reported in the analysis output.

### Fare Regression

Fare prediction is evaluated using:

* MAE
* RMSE
* R²
* Adjusted R²

Residual analysis is also performed to inspect heteroscedasticity. The residual plot indicates increasing residual spread at higher predicted fares.

### Model Persistence

The complete fitted classification pipeline is saved with `joblib`, reloaded, and tested using raw input data.

---

## Module 3 — Zepto Support Assistant

### Architecture

```text
Policy Documents
       ↓
SentenceTransformer
(all-MiniLM-L6-v2)
       ↓
ChromaDB
       ↓
Top-3 Cosine Retrieval
       ↓
LangGraph
       ↓
Intent Classification
   ↙           ↘
Policy       General
Question     Question
   ↓             ↓
Retrieved     Fixed
Context       Response
   ↓
Pydantic Response
       ↓
FastAPI /ask
```

### Policy Documents

The assistant uses eight local Zepto policy documents covering:

1. Delivery
2. Returns and refunds
3. Membership
4. Order tracking
5. Cancellation
6. Damaged or missing items
7. Gift cards
8. Support hours

### Retrieval

Policy documents are embedded locally using `all-MiniLM-L6-v2` and stored in ChromaDB using cosine similarity.

For policy questions, the system always retrieves the top 3 relevant chunks before generating the response.

### LangGraph Workflow

The LangGraph `StateGraph` contains:

* `classify_intent`
* `retrieve_and_answer`
* `direct_answer`

A conditional edge routes policy-related questions to retrieval and general questions to the direct-answer node.

### Mock LLM

`MOCK_LLM` is enabled by default for deterministic local execution.

For policy questions, the mock response follows:

```text
Based on the retrieved context: {top_chunk_snippet}
```

For unrelated questions, the assistant returns a fixed response indicating that it only answers Zepto policy questions.

### Structured Prompt

The optional real-LLM path uses a structured prompt containing:

* Role
* Context
* Task
* Output format
* Response length
* Negative constraints
* Few-shot example

The real-LLM path also supports retries.

### API

FastAPI exposes:

```text
POST /ask
```

Example:

```json
{
  "query": "How can I track my order?"
}
```

The API returns a structured response containing:

```json
{
  "answer": "...",
  "sources": ["doc_04.txt", "doc_06.txt", "doc_01.txt"],
  "confidence": 1
}
```

### Docker

The application is containerized using `support_assistant/Dockerfile`.

Build:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
```

Run:

```bash
docker run -d --name zepto-support-api -p 8000:8000 zepto-support-assistant
```

API documentation:

```text
http://localhost:8000/docs
```

The Dockerized API was tested successfully with the order-tracking query and returned HTTP 200.

---

## Project Structure

```text
zepto-ai-ml-capstone/
│
├── README.md
│
├── data_pipeline/
│   ├── scrape_and_load.py
│   ├── create_database.py
│   ├── run_queries.py
│   ├── books_cleaned.csv
│   ├── books.db
│   └── sql_outputs.txt
│
├── analytics/
│   ├── titanic_analysis.py
│   ├── classification_models.py
│   ├── imbalance_comparison.py
│   ├── random_forest_tuning.py
│   ├── fare_regression.py
│   ├── roc_comparison.py
│   ├── save_and_reload_model.py
│   ├── titanic.csv
│   └── titanic_survival_pipeline.joblib
│
└── support_assistant/
    ├── docs/
    ├── ingest.py
    ├── graph.py
    ├── main.py
    ├── requirements.txt
    ├── Dockerfile
    └── chroma_db/
```

## How to Run

### Data Pipeline

```bash
python data_pipeline/scrape_and_load.py
python data_pipeline/create_database.py
python data_pipeline/run_queries.py
```

### Analytics

```bash
python analytics/titanic_analysis.py
python analytics/classification_models.py
python analytics/imbalance_comparison.py
python analytics/random_forest_tuning.py
python analytics/fare_regression.py
python analytics/roc_comparison.py
python analytics/save_and_reload_model.py
```

### Support Assistant Locally

```bash
python -m uvicorn support_assistant.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### Support Assistant with Docker

```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
docker run -d --name zepto-support-api -p 8000:8000 zepto-support-assistant
```

Then open:

```text
http://localhost:8000/docs
```
