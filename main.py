from googletrans import Translator
import time
import os
from google.cloud import language_v1
import cloudscraper
from bs4 import BeautifulSoup
 
scraper = cloudscraper.create_scraper()
headers = {'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}

URL = "https://www.amazon.com/"
try:
    r = scraper.get(URL, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    title = soup.find('title').text
    description = soup.find('meta', attrs={'name': 'description'})
 
    if "content" in str(description):
        description = description.get("content")
    else:
        description = ""
 
 
    h1 = soup.find_all('h1')
    h1_all = ""
    for x in range (len(h1)):
        if x ==  len(h1) -1:
            h1_all = h1_all + h1[x].text
        else:
            h1_all = h1_all + h1[x].text + ". "
 
 
    paragraphs_all = ""
    paragraphs = soup.find_all('p')
    for x in range (len(paragraphs)):
        if x ==  len(paragraphs) -1:
            paragraphs_all = paragraphs_all + paragraphs[x].text
        else:
            paragraphs_all = paragraphs_all + paragraphs[x].text + ". "
 
 
 
    h2 = soup.find_all('h2')
    h2_all = ""
    for x in range (len(h2)):
        if x ==  len(h2) -1:
            h2_all = h2_all + h2[x].text
        else:
            h2_all = h2_all + h2[x].text + ". "
 
 
 
    h3 = soup.find_all('h3')
    h3_all = ""
    for x in range (len(h3)):
        if x ==  len(h3) -1:
            h3_all = h3_all + h3[x].text
        else:
            h3_all = h3_all + h3[x].text + ". "
 
    allthecontent = ""
    allthecontent = str(title) + " " + str(description) + " " + str(h1_all) + " " + str(h2_all) + " " + str(h3_all) + " " + str(paragraphs_all)
    allthecontent = str(allthecontent)[0:999]
 
except Exception as e:
        print(e)

translator = Translator()
 
try:
        translation = translator.translate(allthecontent).text
        translation = str(translation)[0:999]
        time.sleep(10)
        # print(translation)
        
except Exception as e:
        print(e)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\tiazh\Downloads\verizon-webscraping-test-fa33e5bd3c51.json"
 
try:
        text_content = str(translation)[0:1000]
 
        client = language_v1.LanguageServiceClient()
 
        document = language_v1.Document(
        content=text_content[:1000],  # Truncate to first 1000 characters
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language="en"
    )
        response = client.classify_text(request={'document': document})
        
        # Process and print results
        for category in response.categories:
            print(f"Category: {category.name}")
            print(f"Confidence: {category.confidence:.3%}")

except Exception as e:
        print(e)
 
