import time
import requests
import pandas as pd
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D,SpatialDropout1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import gensim
import sqlite3
import yake


# Scrapping Links
driver=webdriver.Chrome()
driver.get("https://finance.yahoo.com/topic/stock-market-news")
time.sleep(2) 


def execute_times(times):
    for i in range(times + 1):
        driver.execute_script("window.scrollTo(0,1000000000000000000000000000);")
        time.sleep(5)
execute_times(10)


html=driver.page_source
soup=BeautifulSoup(html,'lxml')
url = soup.find_all('a', attrs={'class': 'js-content-viewer wafer-caas Fw(b) Fz(18px) Lh(23px) LineClamp(2,46px) Fz(17px)--sm1024 Lh(19px)--sm1024 LineClamp(2,38px)--sm1024 mega-item-header-link Td(n) C(#0078ff):h C(#000) LineClamp(2,46px) LineClamp(2,38px)--sm1024 not-isInStreamVideoEnabled'})
ls = ['https://finance.yahoo.com/'+i['href'] for i in url]
YahooNews_links = pd.DataFrame(ls)
YahooNews_links.columns = ['links']



# Scrapping Info
def request(url):
    headers = {
        "Host": "finance.yahoo.com",
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
    }
    url = url
    r = requests.get(url,headers=headers)
    r.encoding = r.apparent_encoding
    content = r.text
    return content


def soup(content):
    
    soup = BeautifulSoup(content, "html.parser")
    
    # Abbr
    abbr_temp = soup.find_all(name='div', attrs={'class': 'xray-pill-label'})
    abbr = list(set([abbr_temp[i].string for i in range(1,len(abbr_temp))]))
    abbr = [[i for i in abbr if i != None]]
    
    # Time
    time_temp = soup.find_all(name='time', attrs={'class': 'caas-attr-meta-time'})
    time = [time_temp[0].string]
    
    # Title
    title = [soup.title.text]
    
    # News
    news_temp = soup.find_all(name='div', attrs={'class': 'caas-body'})
    news = [news_temp[0].text]
    
    # Making List
    abbr.append(time)
    abbr.append(title)
    abbr.append(news)

    return abbr


if __name__ == '__main__':
    result = pd.DataFrame()
    
    for i in range(0,len(YahooNews_links)):
        content = request(YahooNews_links.loc[i][0])
        abbr = soup(content)
        
        result = result.append([abbr])
        
    YahooNews_info = result
    YahooNews_info.columns = ['abbr','time','title','news']


YahooNews_info = YahooNews_info.reset_index()
YahooNews_info = pd.concat([YahooNews_info, YahooNews_links], axis=1)
del YahooNews_info['index']


YahooNews_info['abbr'] = [' '.join(i) for i in YahooNews_info['abbr']]
YahooNews_info['time'] = [' '.join(i) for i in YahooNews_info['time']]
YahooNews_info['title'] = [' '.join(i) for i in YahooNews_info['title']]
YahooNews_info = YahooNews_info[~YahooNews_info['abbr'].isin([""])]



# NLP
# 1.Keyword
def key(text):
    
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9 # allowed certain repetition
    numOfKeywords = 20

    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords_temp = custom_kw_extractor.extract_keywords(str(text)) 
    keyword = keywords_temp[0][0]
    
    return keyword


# 2.Attitude
def att(text):
    analyser = SentimentIntensityAnalyzer()
    attitudes = analyser.polarity_scores(text)
    
    if attitudes['neg'] > attitudes['pos']:
        attitude = 'neg'
    else:
        attitude = 'pos'

    return attitude


# 3.Classification
# 3.1 Connectiong
def read_data(file,num=1000000):
    conn = sqlite3.connect(file)
    cursor = conn.cursor()
    sql = f"select * from articles limit {num}"
    cursor.execute(sql)
    result = cursor.fetchall()
    texts = []
    labels = []
    max_len = 0
    for row in result:
        label = row[1]
        text = row[2]
        max_len = max(max_len, len(text.split()))
        texts.append(text)
        labels.append(label)
    assert len(texts) == len(labels)
    return texts, labels

texts, labels = read_data("finance_corpus.db")
set(labels)

# 3.2 Transcoding
encoder = preprocessing.LabelEncoder()
encoder_y = encoder.fit_transform(labels)
encoder_y,set(encoder_y)
y_cate = to_categorical(encoder_y, num_classes=len(set(encoder_y)))

# 3.3 Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)

# 3.4 Sequencing
maxlen = 64
X = pad_sequences(X, padding='post', maxlen=maxlen)

# 3.5 Dividing Train and Test Set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_cate,
                                                    stratify=y_cate,
                                                    random_state=2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 3.6 Word2Vec
sentences = []
for item in texts:
    sentences.append(item.split())
Word2VecModel = gensim.models.Word2Vec(sentences,
                                   vector_size=300,
                                   window=5,
                                   min_count=1,
                                   workers=5)

vocab_list = list(Word2VecModel.wv.key_to_index.keys())

word_index = {" ": 0}
word_vector = {}

embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

for i in range(len(vocab_list)):
    word = vocab_list[i]
    word_index[word] = i + 1
    word_vector[word] = Word2VecModel.wv[word]
    embeddings_matrix[i + 1] = Word2VecModel.wv[word]
vocab_size = len(vocab_list) + 1

# 3.7 CNN Modeling
main_input = Input(shape=(maxlen, ), dtype='float64')

embedder = Embedding(vocab_size, 300, input_length=maxlen, trainable=False)
embed = embedder(main_input)

cnn1 = Convolution1D(256, 3, padding='same', strides=1,
                     activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=4)(cnn1)
cnn2 = Convolution1D(256, 4, padding='same', strides=1,
                     activation='relu')(embed)
cnn2 = MaxPool1D(pool_size=4)(cnn2)
cnn3 = Convolution1D(256, 5, padding='same', strides=1,
                     activation='relu')(embed)
cnn3 = MaxPool1D(pool_size=4)(cnn3)

cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.5)(flat)
main_output = Dense(y_train.shape[1], activation='softmax')(drop)
model = Model(inputs=main_input, outputs=main_output)

# 3.8 Model Training
filepath = 'cnn_weights.best.hdf5'

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint]
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train,
          y=y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_split=0.1,
          callbacks=callbacks_list)

score = model.evaluate(X_test, y_test, verbose=1)
print(score)

# 3.9 Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
model.load_weights(filepath)
preds = np.argmax(model.predict(X_test), axis=-1)
print(f'acc: {accuracy_score(np.argmax(y_test,axis=-1), preds)}')
print(
    f'precision: {precision_score(np.argmax(y_test,axis=-1), preds,average="macro")}'
)
print(
    f'recall: {recall_score(np.argmax(y_test,axis=-1), preds,average="macro")}'
)
print(
    f'f1 score: {f1_score(np.argmax(y_test,axis=-1), preds,average="macro")}')

# 3.10 Labeling
def predict(text, tokenizer, maxlen):
    model.load_weights(filepath)
    texts = [text]
    texts = tokenizer.texts_to_sequences(texts)
    X_test = pad_sequences(texts, padding='post', maxlen=maxlen)
    preds = np.argmax(model.predict(X_test), axis=-1)
    return preds

if __name__ == '__main__':
    result_key = pd.DataFrame()
    result_att = pd.DataFrame()
    result_class = pd.DataFrame()
    
    for i in YahooNews_info['news']:
        keyword = key(i)
        result_key = result_key.append([keyword])
    
    for i in YahooNews_info['news']:
        attitude = att(i)
        result_att = result_att.append([attitude])
        
    for idx, text in YahooNews_info.iterrows():
        pre = predict(text['title'], tokenizer, maxlen)
        result_class = result_class.append([encoder.inverse_transform(pre)])
    
    result = pd.concat([result_key, result_att, result_class], join = 'inner', axis = 1)
    result.columns = ['keyword','attitude','classification']



# Reorganizing Dataframe
result = result.reset_index()
del result['index']

info = YahooNews_info.reset_index()
del info['index']
del info['news']

df = pd.concat([info, result], axis=1)



# Mysql Update
import pymysql.cursors
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer,String


host = "18.116.165.240"
port = "33060"
username = "root"
password = "CBykDq08SZ"
database = "mydb"
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8")


Base = declarative_base()

class Info(Base):
    __tablename__ = "YahooNews"

    abbreviation = Column(String(64), primary_key=True)
    time = Column(String(64))
    title = Column(String(64))
    links = Column(String(64), unique=True)
    keyword = Column(String(64))
    attitude = Column(String(64))
    classification = Column(String(64))

    def __init__(self, abbreviation, time, title, links, keyword, attitude, classification):
        self.abbreviation = abbreviation 
        self.time = time
        self.title = title 
        self.links = links
        self.keyword = keyword 
        self.attitude = attitude
        self.classification = classification 


Base.metadata.create_all(engine)
DbSession = sessionmaker(bind=engine)
session = DbSession()


all_df_temp = engine.execute("SELECT * from YahooNews").fetchall()
all_df = pd.DataFrame(all_df_temp, columns=['abbreviation','time','title','links','keyword','attitude','classification'])
new_df = df.loc[~df['links'].isin(all_df['links'])]
new_df = new_df.reset_index()
del new_df['index']

new_df.to_sql('YahooNews', engine, schema='mydb', index = False, index_label=None, if_exists='append')
session.commit()
session.close()


