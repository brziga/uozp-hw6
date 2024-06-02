import argparse
import json
import gzip
import os
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MultiLabelBinarizer
from lemmagen3 import Lemmatizer
from nltk.corpus import stopwords


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def open_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            print("JSON is valid.")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
    return data

class RTVSlo:

    def __init__(self):

        # scikit-learn one-hot encoder
        self.onehotenc = OneHotEncoder(handle_unknown="ignore")
        
        # scikit-learn multilabelbinarizer
        self.mlb_authors = MultiLabelBinarizer(sparse_output=True) # izhod naj bo sparse matrika
        self.mlb_keywords = MultiLabelBinarizer(sparse_output=True) # izhod naj bo sparse matrika

        # lematizator
        self.lemmatizer = Lemmatizer("sl") # slovenski lematizator

        # stopwordi
        self.stopwords = stopwords.words("slovene") # slovenski stopwordi

        # vektorizator
        self.vectorizer = TfidfVectorizer()

        # model
        self.model = Ridge() 


    def fit(self, train_data: list):
        data = train_data

        # pripravimo podatke - X
        data_part1 = []
        data_authors = []
        data_keywords = []
        data_part2 = []

        for article in data: # iteriramo cez vse clanke
            # samo datum, kot string
            date_string = article["date"].split("T")[0]
            # dan v tednu iz datuma
            day_of_week = datetime.strptime(date_string, "%Y-%m-%d").strftime("%A")

            # (samo) ura objave članka
            hour = article["date"].split("T")[1].split(":")[0]

            # se minute
            minute = round(float(article["date"].split("T")[1].split(":")[1]))
            if minute < 7.5: minute = 0
            elif minute < 22.5: minute = 15
            elif minute < 37.5: minute = 30
            elif minute < 52.5: minute = 45
            elif minute < 60: minute = 59

            #topic
            if "topics" in article.keys():
                # ce ima topic, ga kar direktno vzamemo
                topic = article["topics"] # sem preveril, da je vedno isto, kot v url... - ziher je ziher
            else:
                # ce manjka topic, ga vzamemo iz url-ja
                topic = article["url"].split("/")[3]

            #subtopic (samo prvi)
            subtopic = article["url"].split("/")[4]

            # sestavimo vrstico
            new_row1 = [
                day_of_week,
                hour,
                minute,
                # authors,
                topic,
                subtopic
            ]

            data_part1.append(new_row1)

            # avtorji
            new_row2 = article["authors"]
            data_authors.append(new_row2)

            # keywordi
            new_row3 = article["keywords"]
            data_keywords.append(new_row3)

            # title, lead in tekst
            new_row4 = (article["title"] + 
               " " + 
               article["lead"] + 
               " " + 
               " ".join(article["paragraphs"]))
            data_part2.append(new_row4)




        data_part1_enc = self.onehotenc.fit_transform(data_part1)
        data_authors_onehot = self.mlb_authors.fit_transform(data_authors)
        data_keywords_onehot = self.mlb_keywords.fit_transform(data_keywords)

        data_part2_filtered = []
        for article_line in data_part2:
            tokens = article_line.split()
            lemmatized_words = [self.lemmatizer.lemmatize(token).lower() for token in tokens]
            filtered_words = [word for word in lemmatized_words if word not in self.stopwords]
            new_row = " ".join(filtered_words)
            
            data_part2_filtered.append(new_row)

        data_part2_vect = self.vectorizer.fit_transform(data_part2_filtered)

        data_model = hstack([data_part1_enc, data_authors_onehot, data_keywords_onehot, data_part2_vect])

        # pripravimo y
        ground_truth = []
        for article in data:
            # korenimo st komentarjev
            new_row = np.sqrt(article["n_comments"])
            ground_truth.append(new_row)
        

        # model
        self.model.fit(data_model, ground_truth)


    def predict(self, test_data: list) -> np.array:
        test = test_data
        
        # pripravimo podatke
        test_part1 = []
        test_part2 = []
        test_authors = []
        test_keywords = []

        for article in test:

            # samo datum, kot string
            date_string = article["date"].split("T")[0]
            # dan v tednu iz datuma
            day_of_week = datetime.strptime(date_string, "%Y-%m-%d").strftime("%A")

            # (samo) ura objave članka
            hour = article["date"].split("T")[1].split(":")[0]

            # se minute
            minute = round(float(article["date"].split("T")[1].split(":")[1]))
            if minute < 7.5: minute = 0
            elif minute < 22.5: minute = 15
            elif minute < 37.5: minute = 30
            elif minute < 52.5: minute = 45
            elif minute < 60: minute = 59

            # avtorji
            # authors = " ".join(article["authors"]) # kar zdruzimo v string, da bo encoder lahko sprejel

            #topic
            if "topics" in article.keys():
                # ce ima topic, ga kar direktno vzamemo
                topic = article["topics"] # sem preveril, da je vedno isto, kot v url... - ziher je ziher
            else:
                # ce manjka topic, ga vzamemo iz url-ja
                topic = article["url"].split("/")[3]

            #subtopic (samo prvi)
            subtopic = article["url"].split("/")[4]

            # sestavimo vrstico
            new_row1 = [
                day_of_week,
                hour,
                minute,
                # authors,
                topic,
                subtopic
            ]

            test_part1.append(new_row1)

            # konkateniramo naslov in vse odstavke
            new_row2 = (article["title"] + 
                    " " + 
                    article["lead"] + 
                    " " + 
                    " ".join(article["paragraphs"]))

            test_part2.append(new_row2)

            new_row3 = article["authors"]
            test_authors.append(new_row3)

            new_row5 = article["keywords"]
            test_keywords.append(new_row5)

        # lematizacija in filtriranje za besede besedila
        test_part2_filtered = []
        for article_line in test_part2:
            tokens = article_line.split()
            lemmatized_words = [self.lemmatizer.lemmatize(token).lower() for token in tokens]
            filtered_words = [word for word in lemmatized_words if word not in self.stopwords]
            new_row = " ".join(filtered_words)
            
            test_part2_filtered.append(new_row)

        test_part2 = test_part2_filtered

        # uporabimo isti encoder in vectorizer in mlb-je
        test_part1_enc = self.onehotenc.transform(test_part1)
        test_part2_vect = self.vectorizer.transform(test_part2)
        test_authors_onehot = self.mlb_authors.transform(test_authors)
        test_keywords_onehot = self.mlb_keywords.transform(test_keywords)


        test = hstack([
            test_part1_enc, 
            test_authors_onehot, 
            test_keywords_onehot,
            test_part2_vect])
        
        predictions = self.model.predict(test)
        predictions_submit = []
        for p in predictions:
            # predictions_submit.append(p ** 2)
            pred_squared = np.square(p)
            # predictions_submit.append(round(pred_squared))
            predictions_submit.append(pred_squared)

        return predictions_submit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    args = parser.parse_args()

    train_data = open_json(args.train_data_path)
    test_data = open_json(args.test_data_path)

    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    main()