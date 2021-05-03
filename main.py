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
# Python 3.8.8, scikit-learn 0.24.1, matplotlib 3.3.4
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
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# METRICS
from sklearn.metrics import classification_report
# CLUSTERING
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# CROSS-VALIDATION
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
# UTILITY
import numpy as np
import matplotlib.pyplot as plt

##################################################
# COMPILE REVIEWS
##################################################


# Description: Add reviews of a certain path to specified csv file
# Args:
#   writer              - file to write to
#   path_to_review      - path of reviews to add
#   count               - current count of added reviews
#   classification      - sentiment classification for review
# Return:
#   count               - current count of added reviews
def add_review_to_csv(writer, path_to_review, count, classification):
    for f in os.listdir(path_to_review):
        if f.endswith(".txt"):
            open_file = open(path_to_review + f, "r")
            data = open_file.read()
            writer.writerow([count, f'"{data}"', classification])
            count += 1
            open_file.close()

    return count


# Description: Create a csv file with reviews and sentiment classification
# Args:
#   filename    - name of file to create
#   train       - bool if using reviews from train folder
#   test        - bool if using reviews from test folder
def create_csv_file(filename, train, test):
    header = ['row_number', 'text', 'classification']

    train_path_to_mostly_neg = "labeled_data/train/0_mostly_negative/"
    train_path_to_slightly_neg = "labeled_data/train/1_slightly_negative/"
    train_path_to_neutral = "labeled_data/train/2_neutral/"
    train_path_to_slightly_pos = "labeled_data/train/3_slightly_positive/"
    train_path_to_mostly_pos = "labeled_data/train/4_mostly_positive/"

    test_path_to_mostly_neg = "labeled_data/test/0_mostly_negative/"
    test_path_to_slightly_neg = "labeled_data/test/1_slightly_negative/"
    test_path_to_neutral = "labeled_data/test/2_neutral/"
    test_path_to_slightly_pos = "labeled_data/test/3_slightly_positive/"
    test_path_to_mostly_pos = "labeled_data/test/4_mostly_positive/"

    count = 0

    with open(filename, "w", newline='') as f1:
        writer = csv.writer(f1, delimiter=',')
        writer.writerow(header)
        # add mostly negative reviews
        if train:
            count = add_review_to_csv(writer, train_path_to_mostly_neg, count, 0)
        if test:
            count = add_review_to_csv(writer, test_path_to_mostly_neg, count, 0)

        # add slightly negative reviews
        if train:
            count = add_review_to_csv(writer, train_path_to_slightly_neg, count, 1)
        if test:
            count = add_review_to_csv(writer, test_path_to_slightly_neg, count, 1)

        # add neutral reviews
        if train:
            count = add_review_to_csv(writer, train_path_to_neutral, count, 2)
        if test:
            count = add_review_to_csv(writer, test_path_to_neutral, count, 2)

        # add slightly positive reviews
        if train:
            count = add_review_to_csv(writer, train_path_to_slightly_pos, count, 3)
        if test:
            count = add_review_to_csv(writer, test_path_to_slightly_pos, count, 3)

        # add mostly positive reviews
        if train:
            count = add_review_to_csv(writer, train_path_to_mostly_pos, count, 4)
        if test:
            add_review_to_csv(writer, test_path_to_mostly_pos, count, 4)


print("Compiling reviews into ./generated/imdb_train_test.csv ...\n")
create_csv_file("./generated/imdb_train_test.csv", train=True, test=True)

##################################################
# LOAD IN REVIEWS
##################################################


# Description: Load raw text reviews and classifications from csv file
# Args:
#   filename        - name of csv file to read from
# Return:
#   review_text     - contain review text in preparation for converting them into n-grams.
#   y_train, y_test - contain classification labels for both training and testing our model.
def create_reviews_list(filename):
    review_text = []
    classification = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            review_text.append(row[1])
            classification.append(row[2])
    return review_text, np.array(list(map(int, classification)))

##################################################
# DATA PREPROCESSING
##################################################


print("Loading reviews from ./generated/imdb_train_test.csv ...\n")
review_text, y = create_reviews_list("./generated/imdb_train_test.csv")


# Description: Remove html tags and special characters from raw review text
# Args:
#   review_text     - raw review text to preprocess
# Return:
#   review_text     - preprocessed review text
def preprocess_data(review_text):
    for index, row in enumerate(review_text):
        row = re.sub(r'<.*?>', '', row)
        row = re.sub(r'[^a-zA-Z ]', '', row)
        review_text[index] = row
    return review_text


print("Removing special characters and html tags from loaded data...\n")
review_text = preprocess_data(review_text)

##################################################
# PLOT CLUSTERED DATA WITH FEATURES
##################################################


# Description: Plot clustered features to visualize feature performance
# Args:
#   X   - features
def plot_feature_clusters(X):
    pca = PCA(2)
    kmeans = KMeans(n_clusters=5)
    df = pca.fit_transform(X.toarray())

    # predict labels for clusters
    label = kmeans.fit_predict(df)

    # get unique labels
    u_labels = np.unique(label)

    # plot results
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.legend()
    plt.show()

##################################################
# PREDICTIONS
##################################################


# Description: Predict using 5-fold stratified K-fold cross validation
# Args:
#   X   - features
def cross_validate(X):
    # perform predictions on classifiers
    y_pred = cross_val_predict(SGDClassifier(), X, y)
    print("===============SGD CLASSIFICATION REPORT===============")
    print(classification_report(y, y_pred))

    y_pred = cross_val_predict(svm.SVC(), X, y)
    print("===============SVM CLASSIFICATION REPORT===============")
    print(classification_report(y, y_pred))

    y_pred = cross_val_predict(MultinomialNB(), X, y)
    print("===============NB CLASSIFICATION REPORT===============")
    print(classification_report(y, y_pred))

    y_pred = cross_val_predict(LogisticRegression(), X, y)
    print("===============ME CLASSIFICATION REPORT===============")
    print(classification_report(y, y_pred))
    print("\n\n\n\n")


# FEATURE 1
# Description: Tfidf unigram test with classification report and cross validation
def tfidf_unigram_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(1, 1),
                               max_features=5000)

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# TODO: FEATURE 2
# FEATURE 2
# Description: Add another feature here
def another_test():
    tfidfvec = TfidfVectorizer()  # Use this or CountVectorizer

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# TODO: FEATURE 3
# FEATURE 3
# Description: Add another feature here
def do_another_test():
    tfidfvec = TfidfVectorizer()  # Use this or CountVectorizer

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# run predictions
# Feature 1
print("==========================================")
print("Testing Feature 1: TFIDF W/ UNIGRAMS...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_unigram_test()
# TODO: FEATURE 2
# Feature 2
print("==========================================")
print("Testing Feature 2: {FEATURE NAME}...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
print("N/A")
# another_test()
# TODO: FEATURE 3
# Feature 3
print("==========================================")
print("Testing Feature 3: {FEATURE NAME}...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
print("N/A")
# do_another_test()
# TODO: OTHER FEATURES
print("==========================================")
print("Testing Feature N: {FEATURE NAME}...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
print("N/A")
