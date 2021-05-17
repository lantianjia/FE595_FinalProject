# FE595_FinalProject

Yahoo News Searhing is used for investors to collect news and obtain news nlp analysis in a short time to support their desicions.

Please have a try: 
http://18.116.165.240/

Note:
1. for example, type in ^GSPC to know the recent news of SP500
2. other stock abbreviation also works, eg. INO, WBA
3. showing blank if no data support in the database


File DataUpdate.py is to support update the database. 10 minutes running to scrap approximate recent 100 news from Yahoo Finance.
File Flask.py is running on AWS EC2 instances.



