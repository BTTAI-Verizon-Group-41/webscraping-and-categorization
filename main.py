from googletrans import Translator
import time
import os
from google.cloud import language_v1
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\tiazh\Downloads\verizon-webscraping-test-fa33e5bd3c51.json"


def fetch_web_content(url):
    scraper = cloudscraper.create_scraper()
    headers = {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}

    try:
        # Ensure the URL is properly formatted with http or https
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        r = scraper.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        title = soup.find('title').text
        description = soup.find('meta', attrs={'name': 'description'})
        # print("hellll")

        if "content" in str(description):
            description = description.get("content")
            # print("success")
        else:
            description = ""
            print("failure")

        h1 = soup.find_all('h1')
        # print("h1")
        h1_all = ""
        for x in range(len(h1)):
            if x == len(h1) - 1:
                h1_all = h1_all + h1[x].text
            else:
                h1_all = h1_all + h1[x].text + ". "

        paragraphs_all = ""
        paragraphs = soup.find_all('p')
        for x in range(len(paragraphs)):
            if x == len(paragraphs) - 1:
                paragraphs_all = paragraphs_all + paragraphs[x].text
            else:
                paragraphs_all = paragraphs_all + paragraphs[x].text + ". "

        h2 = soup.find_all('h2')
        h2_all = ""
        for x in range(len(h2)):
            if x == len(h2) - 1:
                h2_all = h2_all + h2[x].text
            else:
                h2_all = h2_all + h2[x].text + ". "

        h3 = soup.find_all('h3')
        h3_all = ""
        for x in range(len(h3)):
            if x == len(h3) - 1:
                h3_all = h3_all + h3[x].text
            else:
                h3_all = h3_all + h3[x].text + ". "

        allthecontent = ""
        allthecontent = str(title) + " " + str(description) + " " + str(h1_all) + " " + str(h2_all) + " " + str(
            h3_all) + " " + str(paragraphs_all)
        allthecontent = str(allthecontent)[0:999]
        # print(allthecontent)

        translator = Translator()

        try:
            translation = translator.translate(allthecontent).text
            translation = str(translation)[0:999]
            time.sleep(10)
            # to avoid hitting limit
            # print(translation)
            return translation

        except Exception as e:
            print(e)
            return allthecontent[:999]
    except Exception as e:
        print(e)
        return None

    """
    breakpoint between functions, below is the sentiment analysis function
    """


def analyze_sentiment(text):
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT,
            language="en"
        )
        response = client.analyze_sentiment(request={'document': document})

        # Process and print results
        sentiment = response.document_sentiment
        print(f"Sentiment Score: {sentiment.score}")  # Sentiment score ranges from -1.0 (negative) to 1.0 (positive)
        print(
            f"Sentiment Magnitude: {sentiment.magnitude}")  # Sentiment magnitude indicates the overall strength of sentiment
        return sentiment.score, sentiment.magnitude

    except Exception as e:
        print(e)
        return None, None


def create_csv(input_csv, output_csv):
    # Load the CSV and initilizing df
    df = pd.read_csv(input_csv, header=None, names=["URL", "Label"])
    df['Sentiment Score'] = None
    df['Sentiment Magnitude'] = None

    # Loop through each row and process the URL
    for index, row in df.head(10).iterrows():
        # try with 10 first
        url = row['URL']
        label = row['Label']  # Assuming your CSV has a column named 'Label'
        print(f"Processing {url} with label '{label}'...")

        # Fetch the web content (already translated)
        web_content = fetch_web_content(url)

        if web_content:
            # Analyze the sentiment of the translated content
            sentiment_score, sentiment_magnitude = analyze_sentiment(web_content)

            if sentiment_score is not None:
                print(f"Sentiment Score for {url}: {sentiment_score}")
                print(f"Sentiment Magnitude for {url}: {sentiment_magnitude}")

                # Store the results in the DataFrame
                df.at[index, 'Sentiment Score'] = sentiment_score
                df.at[index, 'Sentiment Magnitude'] = sentiment_magnitude
            else:
                print(f"Skipping sentiment analysis for {url}")
        else:
            print(f"Skipping {url} due to missing content.")

    # Save the DataFrame back to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# please work im praying
input_csv = "categorizedurls.csv"
output_csv = 'output_with_sentiment.csv'

create_csv(input_csv, output_csv)