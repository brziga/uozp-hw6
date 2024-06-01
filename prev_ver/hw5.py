import argparse
import json
import gzip
import os
import numpy as np

import pandas as pd
from nltk.corpus import stopwords
from lemmagen3 import Lemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def lemmatize_sentence(lemmatizer, sentence):
    tokens = sentence.split()
    words_lemmatized = [lemmatizer.lemmatize(token).lower() for token in tokens]
    sentence_lemmatized = " ".join(words_lemmatized)
    return sentence_lemmatized

def filter_words(text, stopwords, counts, threshold):
    words = text.split()
    if counts is None:
        words_filtered = [w.lower() for w in words if w.lower() not in stopwords and not any(char.isdigit() for char in w)]
    else:
        words_filtered = [w.lower() for w in words if w.lower() not in stopwords and counts[w] >= threshold and not any(char.isdigit() for char in w)]
    text_filtered = " ".join(words_filtered)
    return text_filtered

def filter_words_cntonly(text, counts, threshold):
    words = text.split()
    words_filtered = [w.lower() for w in words if counts[w] >= threshold]
    text_filtered = " ".join(words_filtered)
    return text_filtered


class RTVSlo:

    def __init__(self):
        # Params
        self.desired_dim = 2000 # desired BoW dimension
        self.thresh_count = None
        ########

        # Preparations
        self.sl_stopwords = stopwords.words("slovene") # slovenski stopwordi
        self.lemmatizer = Lemmatizer("sl") # slovenski lematizator
        ##############

    def fit(self, train_data: list):

        # Data
        dataframe = pd.DataFrame(train_data)

        # # fix category and topics columns
        # dataframe.loc[dataframe["topics"].isna(), "topics"] = dataframe["category"]
        # dataframe.loc[dataframe["topics"].isna(), "topics"] = dataframe["url"].apply(lambda x: x.split("/")[3])
        # dataframe.drop("category", axis=1, inplace=True)

        # construct table for model
        data = dataframe[["n_comments"]]
        # concatenate the chosen text columns
        data["concat_text"] = dataframe["title"] + " " + dataframe["lead"] + " " + dataframe["keywords"].apply(lambda x: " ".join(x)) + " " + dataframe["gpt_keywords"].apply(lambda x: " ".join(x)) + " " + dataframe["paragraphs"].apply(lambda x : " ".join(x))

        # lematize and filter
        data["concat_text_lemma"] = data["concat_text"].apply(lambda x: lemmatize_sentence(self.lemmatizer, x))
        data["concat_text_lemma"] = data["concat_text_lemma"].apply(lambda x: " ".join(list(set(x.split()))) )

        data["concat_text_lemma"] = data["concat_text_lemma"].apply(lambda x: filter_words(x, self.sl_stopwords, None, 1))

        concat_text_lemma_texts = data["concat_text_lemma"].to_list()
        counts = {}
        for row in concat_text_lemma_texts:
            words = row.split()
            for w in words:
                if w in counts.keys():
                    counts[w] += 1
                else:
                    counts[w] = 1

        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_counts) <= self.desired_dim:
            self.thresh_count = sorted_counts[-1][1]
        else:
            self.thresh_count = sorted_counts[self.desired_dim][1]

        data["concat_text_lemma"] = data["concat_text_lemma"].apply(lambda x: filter_words_cntonly(x, counts, self.thresh_count))

        # vectorize
        vectorizer = TfidfVectorizer()
        data_vectorized = vectorizer.fit_transform(data["concat_text_lemma"])
        df_tfidf = pd.DataFrame(data_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
        data = pd.concat([data, df_tfidf], axis=1)
        ######

        # Save preprocessing state
        self.vectorizer = vectorizer
        self.feature_names = list(data.columns)

        # Model

        # prepare X and y
        X = data.drop(columns=["n_comments", "concat_text", "concat_text_lemma"])
        y = data["n_comments"].apply(lambda x: np.sqrt(x))

        X_train, y_train = X, y

        # model
        self.model = Ridge(alpha=6.0)
        self.model.fit(X_train, y_train)

        
    def predict(self, test_data: list) -> np.array:

        # Data
        dataframe = pd.DataFrame(test_data)

        # # fix category and topics columns
        # dataframe.loc[dataframe["topics"].isna(), "topics"] = dataframe["category"]
        # dataframe.loc[dataframe["topics"].isna(), "topics"] = dataframe["url"].apply(lambda x: x.split("/")[3])
        # dataframe.drop("category", axis=1, inplace=True)

        # construct table for model
        # data = dataframe.copy() #[["n_comments"]] # new test input does not include comments
        # concatenate the chosen text columns
        dataframe["concat_text"] = dataframe["title"] + " " + dataframe["lead"] + " " + dataframe["keywords"].apply(lambda x: " ".join(x)) + " " + dataframe["gpt_keywords"].apply(lambda x: " ".join(x)) + " " + dataframe["paragraphs"].apply(lambda x : " ".join(x))

        # adjusted to make it work with the new test input
        data = dataframe[["concat_text"]].copy()

        # lematize and filter
        data["concat_text_lemma"] = data["concat_text"].apply(lambda x: lemmatize_sentence(self.lemmatizer, x))
        data["concat_text_lemma"] = data["concat_text_lemma"].apply(lambda x: " ".join(list(set(x.split()))) )

        data["concat_text_lemma"] = data["concat_text_lemma"].apply(lambda x: " ".join([w for w in x.split() if w in self.feature_names]))

        # vectorize
        vectorizer = self.vectorizer
        data_vectorized = vectorizer.transform(data["concat_text_lemma"])
        df_tfidf = pd.DataFrame(data_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
        data = pd.concat([data, df_tfidf], axis=1)
        ######

        # prepare X and y
        # X = data.drop(columns=["n_comments", "concat_text", "concat_text_lemma"])
        X = data.drop(columns=["concat_text", "concat_text_lemma"])
        # y = data["n_comments"]

        # X_test, y_test = X, y
        X_test = X

        y_predictions = self.model.predict(X_test)
        y_predictions = np.square(y_predictions)

        return y_predictions
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    args = parser.parse_args()

    train_data = read_json(args.train_data_path)
    test_data = read_json(args.test_data_path)

    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    main()
