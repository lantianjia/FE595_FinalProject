# Final project in FE595-Financial Technology

# Financial News Search and Sentiment Analysis
This API allows users to search on stock ticker and will return the latest news, categorized and scored based
on sentiment analysis.
Yahoo News Searhing is used for investors to collect news and obtain news nlp analysis in a short time to support their desicions.

There are 4 basic parts to the tool:
1. Scraping tool - webscraping and anti-scraping of Yahoo Finance for news via the following packages:
- BeautifulSoup (bs4)
- webdriver

2. NLP - Senitment analysis, labling and categorization is performed on the scraped data via the following packages:
- Yake
- keras
- tokenizer
- sklearn

3. Flask
File Flask.py is running on AWS EC2 instance.

4. AWS

5. Database
File DataUpdate.py is to support update the database. 10 minutes running to scrap approximate recent 100 news from Yahoo Finance.

# How to use this API:
This API will take in a
1. for example, type in ^GSPC to know the recent news of SP500
2. other stock abbreviation also works, eg. INO, WBA
3. showing blank if no data support in the database

# Call the api via the following URL:
18.116.165.240

&lt;URL&gt;

# Add the following to the base &lt;URL&gt; above to access specific services:

1. &lt;URL&gt;/

    Returns a title page and directions to help.

2.

