import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd

url = "https://books.toscrape.com/"

response = requests.get(url)
response.encoding = "utf-8"

print(response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

books = soup.find_all("article", class_="product_pod")

print("Number of books:", len(books))

def extract_book(book, category):
    # Extract title
    title = book.find("h3").find("a")["title"]

    # Extract and clean price
    price = book.find("p", class_="price_color").text
    price_gbp = float(price.replace("£", "").strip())

    # Extract and clean rating
    rating_element = book.find("p", class_="star-rating")
    rating_text = rating_element.get("class")[1]

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    rating = rating_map[rating_text]

    # Extract and clean availability
    availability = book.find(
        "p", class_="instock availability"
    ).text.strip()

    in_stock = availability == "In stock"

    return {
        "title": title,
        "price_gbp": price_gbp,
        "rating": rating,
        "in_stock": in_stock,
        "category": category
    }

def scrape_category(category_name, category_url):
    current_url = category_url
    category_books = []

    while current_url:
        response = requests.get(current_url)
        response.raise_for_status()
        response.encoding = "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.find_all("article", class_="product_pod")

        for book in books:
            cleaned_book = extract_book(book, category_name)
            category_books.append(cleaned_book)

        next_link = soup.find("li", class_="next")

        if next_link:
            next_href = next_link.find("a")["href"]
            current_url = urljoin(current_url, next_href)
        else:
            current_url = None

    return category_books

mystery_url = (
    "https://books.toscrape.com/catalogue/category/books/"
    "mystery_3/index.html"
)

mystery_books = scrape_category("Mystery", mystery_url)

print("Mystery books:", len(mystery_books))
print(mystery_books[0])

historical_fiction_url = (
    "https://books.toscrape.com/catalogue/category/books/"
    "historical-fiction_4/index.html"
)

historical_fiction_books = scrape_category(
    "Historical Fiction",
    historical_fiction_url
)

print("Historical Fiction books:", len(historical_fiction_books))
print(historical_fiction_books[0])

fiction_url = (
    "https://books.toscrape.com/catalogue/category/books/"
    "fiction_10/index.html"
)

fiction_books = scrape_category(
    "Fiction",
    fiction_url
)

print("Fiction books:", len(fiction_books))
print(fiction_books[0])

all_books = (
    mystery_books
    + historical_fiction_books
    + fiction_books
)

print("Total records:", len(all_books))

df = pd.DataFrame(all_books)

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nDuplicate titles:")
print(df["title"].duplicated().sum())

print("\nRating values:")
print(df["rating"].value_counts().sort_index())

print("\nStock values:")
print(df["in_stock"].value_counts())



print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)


GBP_TO_INR = 105.50

df["price_inr"] = df["price_gbp"] * GBP_TO_INR

print("\nPrice conversion:")
print(df[["price_gbp", "price_inr"]].head())


output_file = "data_pipeline/books_cleaned.csv"

df.to_csv(output_file, index=False)

print("\nDataset saved to:", output_file)
print("Final dataset shape:", df.shape)