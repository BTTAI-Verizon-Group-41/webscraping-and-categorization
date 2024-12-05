import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import requests
from urllib.parse import urlparse
import re

# Accidentally closed loopnet.com, resulting in Selenium error

# List of custom user-agent strings
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
]


# Random delay function
def random_delay(min_delay=8, max_delay=15):
    time.sleep(random.uniform(min_delay, max_delay))


# Function to ensure URLs are formatted properly
def format_url(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        return 'https://' + url
    return url


# Fetch content using requests
def fetch_content_requests(url):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe']):
            tag.decompose()
        clean_text = soup.get_text(separator=" ", strip=True)
        # Remove the error messages
        clean_text = remove_error_messages(clean_text)
        return clean_text
    except RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

# Remove error messages mixed in the scraped contents
def remove_error_messages(text):
    # This only removes the words 'javascript' and 'cookies' (case-insensitive)
    # to reduce the confusion these words might bring to the prediction,
    # but the sentences that contain them will still be in the text.
    text = re.sub(r'javascript|cookies', '', text, flags=re.IGNORECASE)
    return text

# Fetch content using Selenium
def fetch_content_selenium(url):
    options = webdriver.ChromeOptions()
    options.headless = True  # Run in headless mode if you don't need to see the browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")  # Disguise Selenium

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        random_delay()  # Simulate human behavior with a delay
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe']):
            tag.decompose()
        clean_text = soup.get_text(separator=" ", strip=True)
        # Remove the error messages
        clean_text = remove_error_messages(clean_text)
        return clean_text
    except (WebDriverException, TimeoutException) as e:
        print(f"Selenium error for {url}: {e}")
        return None
    finally:
        driver.quit()


def fetch_content(url, use_selenium=False):
    url = format_url(url)
    random_delay()

    # Try requests first
    content = fetch_content_requests(url)
    if content and "access denied" not in content.lower(): # content filter 1
        return content

    # Fallback to Selenium
    if use_selenium:
        return fetch_content_selenium(url)
    return None


def main():
    df = pd.read_csv('tia-nltkmodel/keywords_emptyText.csv')
    results = []
    for index, row in df.iterrows():
        url = row['url']
        content = fetch_content(url, use_selenium=True)

        if content and (content != url): # sometimes when failed to fetch content, content gets the url name
            results.append({"url": url, "content": content})
            print(url + "\n" + content + "\n")
        else:
            results.append({"url": url, "content": None})
            print(f"Failed to retrieve content for {url}")

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('remaining_contents.csv', index=False)

if __name__ == '__main__':
    main()