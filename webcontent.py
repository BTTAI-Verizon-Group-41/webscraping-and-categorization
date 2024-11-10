from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
from googletrans import Translator


# initiate translator
translator = Translator()

# translate text and return original if fails
def translate_to_english(text):
    try:
        translation = translator.translate(text, dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text



#format my urls
def format_url(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        return 'https://' + url 
    return url

# Not yet sure what the header is for but it seems to fix a few issues
def fetch_content(url, use_selenium=False):
    url = format_url(url)  # Ensure URL has a scheme

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    #if the browser is not rendered by js
    if not use_selenium:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as req_err:
            print(f"Failed to fetch {url}: {req_err}")
            return None
    
    #if the browser renders with js, we try to run selenium without the authentication
    else:
        # Run Chrome in headless mode
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems
        options.add_argument('window-size=1920x1080') 
        
        # Initialize WebDriver in headless mode
        driver = webdriver.Chrome(options=options)

        try:
            driver.get(url)
            # Wait until the main content div is loaded; adjust the selector based on target websites
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
        except WebDriverException as web_err:
            print(f"Selenium error for {url}: {web_err}")
            driver.quit()
            return "Requires Selenium"
        finally:
            driver.quit()

    # Remove noisy tags
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'ins', 'iframe']):
        tag.decompose()

    # Extract clean text
    clean_text = soup.get_text(separator=" ", strip=True)
    if clean_text:
        return translate_to_english(clean_text)
    return  clean_text



df = pd.read_csv('unfetched.csv')
selenium_required_urls = []

for index, row in df.iterrows():
    url = row['url']
    clean_text = fetch_content(url, use_selenium=False)
    if not clean_text or "sorry" in (clean_text or "").lower():
        clean_text = fetch_content(url, use_selenium=True)
    
    # If still failed or returned "Requires Selenium", add to selenium_required_urls
    keywords = ["Requires Selenium", "", None, "sorry", "failed", "not found", "404", "access denied", "403", 'temporarily down', 'javascript', 'cookies', 'you are human', 'unusual activity', 'bot', '<script>', 'troubleshoot the problem', '<style', 'cloudflare_error', 'captcha', 'security block', 'access to this page ha been denied', 'privacy error', 'your access has been denied', 'your request has been blocked', 'an error occurred', 'there was a problem', 'js', 'captcha', '{""accountId"":', '']

    # Use any() to check if clean_text matches any condition in the keywords list
    if not clean_text or any(keyword in (clean_text or "").lower() for keyword in keywords[3:]) or clean_text in keywords[:2]:
        selenium_required_urls.append(url)
        # df.at[index, 'text_content'] = "Failed to retrieve content (Requires Selenium)"
    else:
        # Store the content in the DataFrame
        df.at[index, 'text_content'] = clean_text
        
selenium_df = pd.DataFrame(selenium_required_urls, columns=['url'])
   
df.to_csv('from_unfetched.csv', index=False)     
selenium_df.to_csv('urls_requiring_selenium.csv', index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#sample usage of selenium that worked:

# url = "https://amazon.com"
# Fetch the clean content
# clean_text = fetch_content(url, use_selenium=True)  # Set use_selenium=True if needed
# if clean_text:
#     print(clean_text)
