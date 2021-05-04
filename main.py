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

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    ax1.set_title("0 - Mostly Negative")
    ax2.set_title("1 - Slightly Negative")
    ax3.set_title("2 - Neutral")
    ax4.set_title("3 - Slightly Positive")
    ax5.set_title("4 - Mostly Positive")

    # plot results
    for i in u_labels:
        ax1.scatter(df[label == i, 0], df[label == i, 1], label=i, s=50)

    for i in u_labels:
        ax2.scatter(df[label == i, 0], df[label == i, 1], label=i, s=50)

    for i in u_labels:
        ax3.scatter(df[label == i, 0], df[label == i, 1], label=i, s=50)

    for i in u_labels:
        ax4.scatter(df[label == i, 0], df[label == i, 1], label=i, s=50)

    for i in u_labels:
        ax5.scatter(df[label == i, 0], df[label == i, 1], label=i, s=50)

    for i, label in enumerate(y):
        if label == 0:
            ax1.annotate(label, (df[i][0], df[i][1]), xytext=(0, -1), textcoords='offset points', ha='center',
                         va='center', fontsize='11', color='#000000')
        elif label == 1:
            ax2.annotate(label, (df[i][0], df[i][1]), xytext=(0, -1), textcoords='offset points', ha='center',
                         va='center', fontsize='11', color='#000000')
        elif label == 2:
            ax3.annotate(label, (df[i][0], df[i][1]), xytext=(0, -1), textcoords='offset points', ha='center',
                         va='center', fontsize='11', color='#000000')
        elif label == 3:
            ax4.annotate(label, (df[i][0], df[i][1]), xytext=(0, -1), textcoords='offset points', ha='center',
                         va='center', fontsize='11', color='#000000')
        elif label == 4:
            ax5.annotate(label, (df[i][0], df[i][1]), xytext=(0, -1), textcoords='offset points', ha='center',
                         va='center', fontsize='11', color='#000000')

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
                               ngram_range=(1, 1))

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 2
# Description: Count Vectorizer unigram test with classification report and cross validation
def count_vectorizer_unigram_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(1, 1))  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 3
# Description: Tfidf unigram max feature 500 test with classification report and cross validation
def tfidf_unigram_max_feature_500_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(1, 1),
                               max_features=500)
                               

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 4
# Description: Count vectorizer unigram max feature 500 test with classification report and cross validation
def count_vectorizer_unigram_max_feature_500_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(1, 1),
                               max_features=500)  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 5
# Description: Tfidf bigram test with classification report and cross validation
def tfidf_bigram_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(2, 2))

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 6
# Description: Count Vectorizer bigram test with classification report and cross validation
def count_vectorizer_bigram_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(2, 2))  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 7
# Description: Tfidf bigram max feature 500 test with classification report and cross validation
def tfidf_bigram_max_feature_500_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(2, 2),
                               max_features=500)
                               

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 8
# Description: Count vectorizer bigram max feature 500 test with classification report and cross validation
def count_vectorizer_bigram_max_feature_500_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(2, 2),
                               max_features=500)  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 9
# Description: Tfidf trigram test with classification report and cross validation
def tfidf_trigram_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(3, 3))

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 10
# Description: Count Vectorizer trigram test with classification report and cross validation
def count_vectorizer_trigram_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(3, 3))  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 11
# Description: Tfidf trigram max feature 500 test with classification report and cross validation
def tfidf_trigram_max_feature_500_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(3, 3),
                               max_features=500)
                               

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 12
# Description: Count vectorizer trigram max feature 500 test with classification report and cross validation
def count_vectorizer_trigram_max_feature_500_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(3, 3),
                               max_features=500)  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 13
# Description: Tfidf unigram & bigram test with classification report and cross validation
def tfidf_unigram_bigram_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(1, 2))

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 14
# Description: Count Vectorizer unigram & bigram test with classification report and cross validation
def count_vectorizer_unigram_bigram_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(1, 2))  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 15
# Description: Tfidf unigram & bigram max feature 500 test with classification report and cross validation
def tfidf_unigram_bigram_max_feature_500_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(1, 2),
                               max_features=500)
                               

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 16
# Description: Count vectorizer unigram & bigram max feature 500 test with classification report and cross validation
def count_vectorizer_unigram_bigram_max_feature_500_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(1, 2),
                               max_features=500)  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X) 


# FEATURE 17
# Description: Tfidf unigram & bigram & trigram test with classification report and cross validation
def tfidf_unigram_bigram_trigram_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(1, 3))

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 18
# Description: Count Vectorizer unigram & bigram & trigram test with classification report and cross validation
def count_vectorizer_unigram_bigram_trigram_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(1, 3))  

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 19
# Description: Tfidf unigram & bigram & trigram max feature 500 test with classification report and cross validation
def tfidf_unigram_bigram_trigram_max_feature_500_test():
    tfidfvec = TfidfVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               use_idf=True,
                               ngram_range=(1, 3),
                               max_features=500)
                               

    X = tfidfvec.fit_transform(review_text)

    cross_validate(X)

    plot_feature_clusters(X)


# FEATURE 20
# Description: Count vectorizer unigram & bigram & trigram max feature 500 test with classification report and cross validation
def count_vectorizer_unigram_bigram_trigram_max_feature_500_test():
    tfidfvec = CountVectorizer(stop_words="english",
                               analyzer='word',
                               lowercase=True,
                               # use_idf=True,
                               ngram_range=(1, 3),
                               max_features=500)  

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

# Feature 2
print("==========================================")
print("Testing Feature 2: COUNTVECTORIZER W/ UNIGRAMS...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_unigram_test()

# Feature 3
print("==========================================")
print("Testing Feature 3: TFIDF W/ UNIGRAMS 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_unigram_max_feature_500_test()

# Feature 4
print("==========================================")
print("Testing Feature 4: COUNTVECTORIZER W/ UNIGRAMS 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_unigram_max_feature_500_test()

# Feature 5
print("==========================================")
print("Testing Feature 5: TFIDF W/ BIGRAMS...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_bigram_test()

# Feature 6
print("==========================================")
print("Testing Feature 6: COUNTVECTORIZER W/ BIGRAMS...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_bigram_test()

# Feature 7
print("==========================================")
print("Testing Feature 7: TFIDF W/ BIGRAMS 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_bigram_max_feature_500_test()

# Feature 8
print("==========================================")
print("Testing Feature 8: COUNTVECTORIZER W/ BIGRAMS 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_bigram_max_feature_500_test()

# Feature 9
print("==========================================")
print("Testing Feature 9: TFIDF W/ TRIGRAMS...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_trigram_test()

# Feature 10
print("==========================================")
print("Testing Feature 10: COUNTVECTORIZER W/ TRIGRAMS...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_trigram_test()

# Feature 11
print("==========================================")
print("Testing Feature 11: TFIDF W/ TRIGRAMS 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_trigram_max_feature_500_test()

# Feature 12
print("==========================================")
print("Testing Feature 12: COUNTVECTORIZER W/ TRIGRAMS 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_trigram_max_feature_500_test()

# Feature 13
print("==========================================")
print("Testing Feature 13: TFIDF W/ UNIGRAMS/BIGRAM...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_unigram_bigram_test()

# Feature 14
print("==========================================")
print("Testing Feature 14: COUNTVECTORIZER W/ UNIGRAMS/BIGRAM...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_unigram_bigram_test()

# Feature 15
print("==========================================")
print("Testing Feature 15: TFIDF W/ UNIGRAMS/BIGRAM 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_unigram_bigram_max_feature_500_test()

# Feature 16
print("==========================================")
print("Testing Feature 16: COUNTVECTORIZER W/ UNIGRAMS/BIGRAM 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_unigram_bigram_max_feature_500_test()

# Feature 17
print("==========================================")
print("Testing Feature 17: TFIDF W/ UNIGRAMS/BIGRAM/TRIGRAM...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_unigram_bigram_trigram_test()

# Feature 18
print("==========================================")
print("Testing Feature 18: COUNTVECTORIZER W/ UNIGRAMS/BIGRAM/TRIGRAM...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_unigram_bigram_trigram_test()

# Feature 19
print("==========================================")
print("Testing Feature 19: TFIDF W/ UNIGRAMS/BIGRAM/TRIGRAM 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
tfidf_unigram_bigram_trigram_max_feature_500_test()

# Feature 20
print("==========================================")
print("Testing Feature 20: COUNTVECTORIZER W/ UNIGRAMS/BIGRAM/TRIGRAM 500 MAX...")
print("(Using stratified 5-fold cross-validation)")
print("==========================================\n")
count_vectorizer_unigram_bigram_trigram_max_feature_500_test()