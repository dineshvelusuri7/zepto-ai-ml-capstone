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