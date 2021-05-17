# Final project in FE595-Financial Technology
# Group 1
- Lantian (Laney) Jia
- Wenbo (Brian) Yu
- Daniel (Dan) Bachalis

# Responsibilities:
- Lantian was responsible for NLP analysis and hosted on AWS
- Wenbo was responsible for the flask API and HTML code
- Dan was responsible for scraping Yahoo News and updating the database in SQL
- All were responsible for reviewing each other's code, testing and troubleshooting

# Financial News Search and Sentiment Analysis API
This API allows users to search on stock ticker and will return the latest news, categorized and scored based
on sentiment analysis.
The goal is to help investors quickly identify important news for their investment decisions.

There are 5 basic parts to the tool:
1. Scraping tool - webscraping and anti-scraping of Yahoo Finance for news via the following packages:
- BeautifulSoup (bs4)
- webdriver

2. NLP - Sentiment analysis, labeling and categorization is performed on the scraped data via the following packages:
- Yake
- keras
- sklearn
- vaderSentiment.vaderSentiment

3. Flask - API created for user interaction and display of NLP results via the following packages:
- flask
- pymysql
- sqlalchemy

4. AWS - This API is running on an AWS EC2 (nano) instance.
- Flask.py is the specific file which is executed and running.

5. Database - SQL database that contains results from web scraping Yahoo Finance.
- File DataUpdate.py is run periodically (10 min runtime) to update the database with additional news.
- Runtime is ~10 min and will scrap approximately 100 of the most recent news listings from Yahoo Finance.
- The database currently only consists of a few hundred records, however it will grow over time as we continue to update.
- Created via the sqlite3 package.

# How to use this API:

Call the API via the following URL:

18.116.165.240

# Add the following to the base &lt;URL&gt; above to access additional services:

1. &lt;URL&gt;

On the main screen, enter a ticker symbol into the search field and click [submit] or press 'Enter' on the keyboard.
The API will return relevant entries for the given symbol along with the following:
- Abbreviation: i.e. ticker symbol
- Time: Date/Time of the article
- Link: A URL to the news article that can be copy and pasted into a browser for viewing the article
- Keyword: Any keywords that the NLP module identified for an given article
- Attitude: The sentiment analysis score for the article (pos = positive sentiment, neg = negative sentiment)
- Classification: The category assigned to the article from the NLP module (e.g. Commodities, Economy, Policy, etc.)

For example, type in ^GSPC to know the recent news for the S&P 500.

NOTE: A search will return a blank list if there is currently no data for a given symbol in the database.
Below are some good stock abbreviations to test this API:
- ^GSPC
- COMP
- GOOGL
- INO
- WBA
- AAPL ETSY

2. &lt;URL&gt;/help

Displays this README file.
