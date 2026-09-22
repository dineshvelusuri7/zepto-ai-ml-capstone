import sqlite3
import pandas as pd


database_file = "data_pipeline/books.db"

connection = sqlite3.connect(database_file)


queries = {
    "Query 1 - SELECT and WHERE": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating = 5;
    """,

    "Query 2 - ORDER BY and LIMIT": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    "Query 3 - DISTINCT": """
        SELECT DISTINCT category_name
        FROM categories;
    """,

    "Query 4 - IN": """
        SELECT title, category_id
        FROM books
        WHERE category_id IN (1, 3);
    """,

    "Query 5 - BETWEEN": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40;
    """,

    "Query 6 - JOIN": """
        SELECT
            books.title,
            books.price_gbp,
            books.rating,
            categories.category_name
        FROM books
        JOIN categories
            ON books.category_id = categories.category_id;
    """
}


output_file = "data_pipeline/sql_outputs.txt"

with open(output_file, "w", encoding="utf-8") as file:

    for query_name, query in queries.items():

        print("\n" + "=" * 60)
        print(query_name)
        print("=" * 60)

        print(query.strip())

        file.write("\n" + "=" * 60 + "\n")
        file.write(query_name + "\n")
        file.write("=" * 60 + "\n")
        file.write(query.strip() + "\n\n")

        result = pd.read_sql(query, connection)

        print(result.head(10))

        file.write(result.head(10).to_string(index=False))
        file.write("\n")


print("\nSQL queries completed.")
print("Output saved to:", output_file)


books_df = pd.read_sql(
    "SELECT * FROM books",
    connection
)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    connection
)

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

print("\nPandas merge result:")
print(
    merged_df[
        ["title", "price_gbp", "rating", "category_name"]
    ].head()
)

connection.close()