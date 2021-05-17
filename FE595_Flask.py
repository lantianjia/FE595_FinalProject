import pymysql.cursors
from sqlalchemy import create_engine
from flask import Flask, render_template, request
import csv

# AWS Database Connection
host = "18.116.165.240"
port = "33060"
username = "root"
password = "CBykDq08SZ"
database = "mydb"
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8")


from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
Base.metadata.create_all(engine)
DbSession = sessionmaker(bind=engine)
session = DbSession()


# Flask
app = Flask(__name__)

def read_sql(sql):
    res_list = []
    for row in sql.fetchall():
        res_list.append(row)
    return res_list

def search(content):
    search_res = []
    search_res.append(res_list[0])
    for item in res_list:
        if content == item[0]:
            search_res.append(item)
    return search_res

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/news', methods=['get', 'post'])
def result():
    content = request.form.get('content')
    search_res = search(content=content)
    return render_template('index.html',result=search_res)

if __name__ == '__main__':
    res_list = read_sql(engine.execute("SELECT * from YahooNews"))
    app.run(host='0.0.0.0', port=80)

















