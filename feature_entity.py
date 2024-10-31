# pip install selenium (applicable to dynamic websites)
# Modify path_to_your_chromedriver
# Didn't test whether this piece of code would work
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from google.cloud import language_v1
import os

# Setup for Selenium WebDriver
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (without opening a browser window)
    service = Service(executable_path="path_to_your_chromedriver")  # Path to your ChromeDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


# Function to analyze entities using Google Cloud Natural Language API
def analyze_entities(text):
    client = language_v1.LanguageServiceClient()

    # Prepare the document to analyze
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )

    # Analyze the text for entities
    response = client.analyze_entities(document=document)

    entities = []
    for entity in response.entities:
        entities.append({
            "name": entity.name,
            "type": language_v1.Entity.Type(entity.type_).name,
            "salience": entity.salience,
        })

    return entities


# Scrape product details from a webpage using Selenium
def scrape_amazon_selenium(url):
    driver = setup_driver()
    driver.get(url)

    # Wait for the page to load completely
    time.sleep(3)

    # Try to get the product title
    try:
        title = driver.find_element(By.ID, "productTitle").text
        print(f"Product Title: {title}")
    except Exception as e:
        print(f"Error fetching product title: {e}")
        title = None

    # Try to get the product description
    try:
        description = driver.find_element(By.ID, "productDescription").text
        print(f"Product Description: {description}")
    except Exception as e:
        print(f"Error fetching product description: {e}")
        description = None

    driver.quit()

    # Combine title and description for analysis
    full_text = f"Title: {title}\nDescription: {description}"

    return full_text


# Main function
def main():
    # Set the path to your Google Cloud service account key (JSON)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'verizon-web-scraping-ddc9ca44aa31.json'

    # Example product URL from Amazon (modify this URL as needed)
    amazon_url = 'https://www.amazon.com'  # Replace with your target product URL

    # Step 1: Scrape product details from Amazon using Selenium
    scraped_text = scrape_amazon_selenium(amazon_url)

    if scraped_text:
        # Step 2: Analyze the scraped text with Google Cloud Natural Language API
        entities = analyze_entities(scraped_text)

        # Step 3: Output the extracted entities
        print("\nEntities found in the text:")
        for entity in entities:
            print(f"Name: {entity['name']}, Type: {entity['type']}, Salience: {entity['salience']}")
    else:
        print("Failed to scrape any content.")


if __name__ == '__main__':
    main()