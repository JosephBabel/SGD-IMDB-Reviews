#!/usr/bin/env python
# coding: utf-8

##################################################
# We build several models using different classification algorithms in conjunction with n-gram and tf-idf features
# to find the best combination for classifying IMDB reviews into multiple sentiment classes.
##################################################
# Authors: Joseph Babel, Cameron Harte
##################################################
# Task Distribution:
# 
# Joseph Babel: Preprocess data, train and test SGD classifier, train and test SVM classifier,
# build confusion matrix with better prediction metrics
# 
# Cameron Harte: Build csv files for train and test data, build ngram models, train and test NB classifier,
# train and test ME classifier
##################################################
# **Required Modules:**
# Python 3.8, scikit-learn 0.24
##################################################

# FILE I/O
import csv
import os
# PREPROCESSING
import re
# FEATURE EXTRACTOR
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# CLASSIFIERS
from sklearn.linear_model import SGDClassifier
# METRICS
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


def create_csv_file(filename, train):
    header = ['row_number', 'text', 'classification']

    if train:
        path_to_mostly_neg = "labeled_data/train/0_mostly_negative/"
        path_to_slightly_neg = "labeled_data/train/1_slightly_negative/"
        path_to_neutral = "labeled_data/train/2_neutral/"
        path_to_slightly_pos = "labeled_data/train/3_slightly_positive/"
        path_to_mostly_pos = "labeled_data/train/4_mostly_positive/"
    else:
        path_to_mostly_neg = "labeled_data/test/0_mostly_negative/"
        path_to_slightly_neg = "labeled_data/test/1_slightly_negative/"
        path_to_neutral = "labeled_data/test/2_neutral/"
        path_to_slightly_pos = "labeled_data/test/3_slightly_positive/"
        path_to_mostly_pos = "labeled_data/test/4_mostly_positive/"

    count = 0

    with open(filename, "w", newline='') as f1:
        writer = csv.writer(f1, delimiter=',')
        writer.writerow(header)
        # add mostly negative reviews
        for f in os.listdir(path_to_mostly_neg):
            classification = 0
            if f.endswith(".txt"):
                open_file = open(path_to_mostly_neg + f, "r")
                data = open_file.read()
                writer.writerow([count, f'"{data}"', classification])
                count += 1
                open_file.close()
        # add slightly negative reviews
        for f in os.listdir(path_to_slightly_neg):
            classification = 1
            if f.endswith(".txt"):
                open_file = open(path_to_slightly_neg + f, "r")
                data = open_file.read()
                writer.writerow([count, f'"{data}"', classification])
                count += 1
                open_file.close()
        # add neutral reviews
        for f in os.listdir(path_to_neutral):
            classification = 2
            if f.endswith(".txt"):
                open_file = open(path_to_neutral + f, "r")
                data = open_file.read()
                writer.writerow([count, f'"{data}"', classification])
                count += 1
                open_file.close()
        # add slightly positive reviews
        for f in os.listdir(path_to_slightly_pos):
            classification = 3
            if f.endswith(".txt"):
                open_file = open(path_to_slightly_pos + f, "r")
                data = open_file.read()
                writer.writerow([count, f'"{data}"', classification])
                count += 1
                open_file.close()
        # add mostly positive reviews
        for f in os.listdir(path_to_mostly_pos):
            classification = 4
            if f.endswith(".txt"):
                open_file = open(path_to_mostly_pos + f, "r")
                data = open_file.read()
                writer.writerow([count, f'"{data}"', classification])
                count += 1
                open_file.close()


print("Compiling reviews into ./generated/imdb_train.csv, ./generated/imdb_test.csv ...\n")
create_csv_file("./generated/imdb_train.csv", train=True)
create_csv_file("./generated/imdb_test.csv", train=False)


# ## Create List of Reviews and Classifications
# review_text_train, review_text_test - contain review text in preparation for converting them into n-grams.
# y_train, y_test - contain classification labels for both training and testing our model.
def create_reviews_list(filename):
    review_text = []
    classification = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            review_text.append(row[1])
            classification.append(row[2])
    return review_text, classification


print("Loading reviews from ./generated/imdb_train.csv, ./generated/imdb_test.csv ...\n")
review_text_train, y_train = create_reviews_list("./generated/imdb_train.csv")
review_text_test, y_test = create_reviews_list("./generated/imdb_test.csv")

# convert classifications to list of integers
y_train = list(map(int, y_train))
y_test = list(map(int, y_test))


# ## Preprocess Data
# remove html tags and special characters
def preprocess_data(review_text):
    for index, row in enumerate(review_text):
        row = re.sub(r'<.*?>', '', row)
        row = re.sub(r'[^a-zA-Z. ]', '', row)
        review_text[index] = row
    return review_text


print("Removing special characters and html tags from loaded data...\n")
review_text_train = preprocess_data(review_text_train)
review_text_test = preprocess_data(review_text_test)


# ## Convert Review Text Into N-Grams
def text_to_ngram(review_text_train, review_text_test, ngram_range, tfidf):
    if tfidf:
        # use_idf when 'True' enables inverse-document-frequency re-weighting
        # ngram_range = ngram_range sets the lower and upper boundary of range of n-values
        tfidfvec = TfidfVectorizer(stop_words="english",
                                   analyzer='word',
                                   lowercase=True,
                                   use_idf=True,
                                   ngram_range=ngram_range)

        # training data learns vocabulary dictionary and returns document-term matrix 
        x_train = tfidfvec.fit_transform(review_text_train)

        # transforms the test data to document-term matrix
        x_test = tfidfvec.transform(review_text_test)
    else:
        cvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               ngram_range=ngram_range)

        x_train = cvec.fit_transform(review_text_train)

        x_test = cvec.transform(review_text_test)

    return x_train, x_test


# ## SGD Classifier (Stochastic Gradient Descent)
def sgd_classifier(review_text_train, y_train, review_text_test, ngram_range, tfidf):
    x_train, x_test = text_to_ngram(review_text_train, review_text_test, ngram_range, tfidf)

    clf = SGDClassifier(loss="hinge", penalty="l1")

    clf.fit(x_train, y_train)

    prediction = clf.predict(x_test)

    return prediction


# ## SVM Classifier (Support Vector Machine)
def svm_classifier(review_text_train, y_train, review_text_test, ngram_range, tfidf):
    x_train, x_test = text_to_ngram(review_text_train, review_text_test, ngram_range, tfidf)


# ## NB Classifier (Naive Bayes)
def nb_classifier(review_text_train, y_train, review_text_test, ngram_range, tfidf):
    x_train, x_test = text_to_ngram(review_text_train, review_text_test, ngram_range, tfidf)

    clf = MultinomialNB()

    clf.fit(x_train, y_train)

    prediction = clf.predict(x_test)

    return prediction


# ## ME Classifier (Maximum Entropy)
def me_classifier(review_text_train, y_train, review_text_test, ngram_range, tfidf):
    x_train, x_test = text_to_ngram(review_text_train, review_text_test, ngram_range, tfidf)


# # Predictions
# ## SGD Classifier
print("Training and predicting Stochastic Gradient Descent Classifier...\n")
y_pred_unigram = sgd_classifier(review_text_train, y_train, review_text_test, (1, 1), False)
y_pred_bigram = sgd_classifier(review_text_train, y_train, review_text_test, (2, 2), False)
y_pred_trigram = sgd_classifier(review_text_train, y_train, review_text_test, (3, 3), False)
y_pred_unigram_bigram = sgd_classifier(review_text_train, y_train, review_text_test, (1, 2), False)
y_pred_bigram_trigram = sgd_classifier(review_text_train, y_train, review_text_test, (2, 3), False)
y_pred_unigram_bigram_trigram = sgd_classifier(review_text_train, y_train, review_text_test, (1, 3), False)

y_pred_unigram_tfidf = sgd_classifier(review_text_train, y_train, review_text_test, (1, 1), True)
y_pred_bigram_tfidf = sgd_classifier(review_text_train, y_train, review_text_test, (2, 2), True)
y_pred_trigram_tfidf = sgd_classifier(review_text_train, y_train, review_text_test, (3, 3), True)
y_pred_unigram_bigram_tfidf = sgd_classifier(review_text_train, y_train, review_text_test, (1, 2), True)
y_pred_bigram_trigram_tfidf = sgd_classifier(review_text_train, y_train, review_text_test, (2, 3), True)
y_pred_unigram_bigram_trigram_tfidf = sgd_classifier(review_text_train, y_train, review_text_test, (1, 3), True)

print("SGD Classifier Accuracy Scores:")
print("Unigram:\t\t\t\t" + str(accuracy_score(y_test, y_pred_unigram)))
print("Bigram:\t\t\t\t\t" + str(accuracy_score(y_test, y_pred_bigram)))
print("Trigram:\t\t\t\t" + str(accuracy_score(y_test, y_pred_trigram)))
print("Unigram + Bigram:\t\t\t" + str(accuracy_score(y_test, y_pred_unigram_bigram)))
print("Bigram + Trigram:\t\t\t" + str(accuracy_score(y_test, y_pred_bigram_trigram)))
print("Unigram + Bigram + Trigram:\t\t" + str(accuracy_score(y_test, y_pred_unigram_bigram_trigram)) + "\n")

print("Unigram w/ tf-idf:\t\t\t" + str(accuracy_score(y_test, y_pred_unigram_tfidf)))
print("Bigram w/ tf-idf:\t\t\t" + str(accuracy_score(y_test, y_pred_bigram_tfidf)))
print("Trigram w/ tf-idf:\t\t\t" + str(accuracy_score(y_test, y_pred_trigram_tfidf)))
print("Unigram + Bigram w/ tf-idf:\t\t" + str(accuracy_score(y_test, y_pred_unigram_bigram_tfidf)))
print("Bigram + Trigram w/ tf-idf:\t\t" + str(accuracy_score(y_test, y_pred_bigram_trigram_tfidf)))
print(
    "Unigram + Bigram + Trigram w/ tf-idf:\t" + str(accuracy_score(y_test, y_pred_unigram_bigram_trigram_tfidf)) + "\n")

# ## NB Classifier
print("Training and predicting Naive Bayes Classifier ...\n")
y_pred_unigram = nb_classifier(review_text_train, y_train, review_text_test, (1, 1), False)
y_pred_bigram = nb_classifier(review_text_train, y_train, review_text_test, (2, 2), False)
y_pred_trigram = nb_classifier(review_text_train, y_train, review_text_test, (3, 3), False)
y_pred_unigram_bigram = nb_classifier(review_text_train, y_train, review_text_test, (1, 2), False)
y_pred_bigram_trigram = nb_classifier(review_text_train, y_train, review_text_test, (2, 3), False)
y_pred_unigram_bigram_trigram = nb_classifier(review_text_train, y_train, review_text_test, (1, 3), False)

y_pred_unigram_tfidf = nb_classifier(review_text_train, y_train, review_text_test, (1, 1), True)
y_pred_bigram_tfidf = nb_classifier(review_text_train, y_train, review_text_test, (2, 2), True)
y_pred_trigram_tfidf = nb_classifier(review_text_train, y_train, review_text_test, (3, 3), True)
y_pred_unigram_bigram_tfidf = nb_classifier(review_text_train, y_train, review_text_test, (1, 2), True)
y_pred_bigram_trigram_tfidf = nb_classifier(review_text_train, y_train, review_text_test, (2, 3), True)
y_pred_unigram_bigram_trigram_tfidf = nb_classifier(review_text_train, y_train, review_text_test, (1, 3), True)

print("NB Classifier Accuracy Scores:")
print("Unigram:\t\t\t\t" + str(accuracy_score(y_test, y_pred_unigram)))
print("Bigram:\t\t\t\t\t" + str(accuracy_score(y_test, y_pred_bigram)))
print("Trigram:\t\t\t\t" + str(accuracy_score(y_test, y_pred_trigram)))
print("Unigram + Bigram:\t\t\t" + str(accuracy_score(y_test, y_pred_unigram_bigram)))
print("Bigram + Trigram:\t\t\t" + str(accuracy_score(y_test, y_pred_bigram_trigram)))
print("Unigram + Bigram + Trigram:\t\t" + str(accuracy_score(y_test, y_pred_unigram_bigram_trigram)) + "\n")

print("Unigram w/ tf-idf:\t\t\t" + str(accuracy_score(y_test, y_pred_unigram_tfidf)))
print("Bigram w/ tf-idf:\t\t\t" + str(accuracy_score(y_test, y_pred_bigram_tfidf)))
print("Trigram w/ tf-idf:\t\t\t" + str(accuracy_score(y_test, y_pred_trigram_tfidf)))
print("Unigram + Bigram w/ tf-idf:\t\t" + str(accuracy_score(y_test, y_pred_unigram_bigram_tfidf)))
print("Bigram + Trigram w/ tf-idf:\t\t" + str(accuracy_score(y_test, y_pred_bigram_trigram_tfidf)))
print(
    "Unigram + Bigram + Trigram w/ tf-idf:\t" + str(accuracy_score(y_test, y_pred_unigram_bigram_trigram_tfidf)) + "\n")
