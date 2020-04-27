# Load libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import collections
import operator

# Import text data
raw_training = pd.read_csv("training.csv")
raw_testing = pd.read_csv("test.csv")

# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(raw_training["article_words"])

# Create feature matrix
X = bag_of_words

# Create bag of words
y = raw_training["topic"]

#######################Resampling Dataset#######################


# Reducing the effect imbalnced by deleting some irrelevant class
# "Irrelevant" classe has 4734 samples in the training data, try to reduce it into 2000
irrelevant = raw_training[raw_training["topic"] == "IRRELEVANT"]
remove_n = 2734
drop_indices = np.random.choice(irrelevant.index, remove_n, replace=False)
irrelevant = irrelevant.drop(drop_indices)

reduce_training =  pd.concat([raw_training[raw_training["topic"] != "IRRELEVANT"], irrelevant],ignore_index=True)
reduce_bag_of_words = count.fit_transform(reduce_training["article_words"])
R_X = reduce_bag_of_words
R_y = reduce_training["topic"]

# Icreasing the minor classes
# Increasing 
topic_class = raw_training[raw_training["topic"] != "IRRELEVANT"]
increase_training = pd.concat([topic_class, topic_class, topic_class, raw_training[raw_training["topic"] == "IRRELEVANT"]], ignore_index=True)
increase_bag_of_words = count.fit_transform(increase_training["article_words"])
I_X = increase_bag_of_words
I_y = increase_training["topic"]

# Considering stop words
count2 = CountVectorizer(stop_words='english')
sw_bag_of_words = count.fit_transform(increase_training["article_words"])
s_X = sw_bag_of_words
s_y = increase_training["topic"]

sw_bag_of_words

y

def Model_Score (X, y, method, k):
    clf = method
    
    accuracy_scores = cross_val_score(clf, X, y, cv=k, scoring="accuracy")
    precision_scores = cross_val_score(clf, X, y, cv=k, scoring="precision_macro")
    recall_scores = cross_val_score(clf, X, y, cv=k, scoring="recall_macro")
    f1_scores = cross_val_score(clf, X, y, cv=k, scoring="f1_macro")
    
    return np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)
	
# creating 10 fold for k-fold validation
#A = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [5, 6], [7, 2], [5, 4], [7, 5], [10, 4]]) 
b = np.array([1, 2, 3, 4,5,6,7,8,9,10])

# k-fold split number 10 
k = 10

# without doing any data cleaning

bernoulliNB_accuracy, bernoulliNB_precision, bernoulliNB_recall, bernoulliNB_f1 = Model_Score(X, y, BernoulliNB(), 10)
multinomialNB_accuracy, multinomialNB_precision, multinomialNB_recall, multinomialNB_f1 = Model_Score(X, y, MultinomialNB(), 10)


# multinomialNB using uniformed distribution
multiNB_accuracy2, multiNB_precision2, multiNB_recall2, multiNB_f12 = Model_Score(X, y, MultinomialNB(fit_prior = False), 10)


# reduce irrelevant samples
R_accuracy, R_precision, R_recall, R_f1 = Model_Score(R_X, R_y, MultinomialNB(), 10)


# Icreasing the minor classes
I_accuracy, I_precision, I_recall, I_f1 = Model_Score(I_X, I_y, MultinomialNB(), 10)


# Try considering stop words
s_accuracy, s_precision, s_recall, s_f1 = Model_Score(s_X, s_y, MultinomialNB(), 10)



vectorizer = TfidfVectorizer()
#tf-idf 
tfidf = vectorizer.fit_transform(raw_training["article_words"])
tfidf0_X = tfidf
tfidf0_y = raw_training["topic"]
# score using tf-idf words extraction
tfidf0_accuracy, tfidf0_precision, tfidf0_recall, tfidf0_f1 = Model_Score(tfidf0_X, tfidf0_y, MultinomialNB(), 10)
#tf-idf with increased data
increase_tfidf = vectorizer.fit_transform(increase_training["article_words"])
tfidf_X = increase_tfidf
tfidf_y = increase_training["topic"]
# score using tf-idf words extraction
tfidf_accuracy, tfidf_precision, tfidf_recall, tfidf_f1 = Model_Score(tfidf_X, tfidf_y, MultinomialNB(), 10)


# Try Classification report
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

clf_BernoulliNB = BernoulliNB()
model_BernoulliNB = clf_BernoulliNB.fit(X_train, y_train)

clf_MultinomialNB = MultinomialNB()
model_MultinomialNB = clf_MultinomialNB.fit(X_train, y_train)


predicted_BernoulliNB = model_BernoulliNB.predict(X_valid)
predicted_MultinomialNB = model_MultinomialNB.predict(X_valid)

I_X_train, I_X_valid, I_y_train, I_y_valid = train_test_split(I_X, I_y, test_size=0.20, random_state=42)
model_MultinomialNB_2 = clf_MultinomialNB.fit(I_X_train, I_y_train)
predicted_MultinomialNB_2 = model_MultinomialNB_2.predict(I_X_valid)


# still need to deal with 


# 1. irrelevant articles

# 2. The distribution of topics are not uniform

# 3. select the features

# 4. maybe can give penalty to the misclassifying




# multinomialNB_accuracy = Model_Score(X, y, MultinomialNB(), 10, "accuracy")

print("====================================================================================")
print("=============================using bag of words=====================================")
print("====================================================================================")
print("Without doing any data cleaning, the score of bernoulliNB,\naccuracy:  " + str(bernoulliNB_accuracy) +
     "\nprecision: " + str(bernoulliNB_precision) + "\nrecall:    " + str(bernoulliNB_recall) + "\nf1:        " +
     str(bernoulliNB_f1))
#print("\nClassification Report for bernoulliNB:\n")
#print(classification_report(y_valid, predicted_BernoulliNB))

print("====================================================================================")

print("Without doing any data cleaning, the score of multinomialNB,\naccuracy:  " + str(multinomialNB_accuracy) +
     "\nprecision: " + str(multinomialNB_precision) + "\nrecall:    " + str(multinomialNB_recall) + "\nf1:        " +
     str(multinomialNB_f1))
#print("\nClassification Report for multinomialNB:\n")
#print(classification_report(y_valid, predicted_MultinomialNB))

print("====================================================================================")

print("Setting uniformed prior, the score of multinomialNB,\naccuracy:  " + str(multiNB_accuracy2) +
     "\nprecision: " + str(multiNB_precision2) + "\nrecall:    " + str(multiNB_recall2) + "\nf1:        " +
     str(multiNB_f12))
#print("\nClassification Report for multinomialNB:\n")
#print(classification_report(y_valid, predicted_MultinomialNB))

print("====================================================================================")

print("Reduce the case of irrelevant, the score of multinomialNB,\naccuracy:  " + str(R_accuracy) +
     "\nprecision: " + str(R_precision) + "\nrecall:    " + str(R_recall) + "\nf1:        " +
     str(R_f1))

print("====================================================================================")

print("Increase(Copy) the case of topic classes, the score of multinomialNB,\naccuracy:  " + str(I_accuracy) +
     "\nprecision: " + str(I_precision) + "\nrecall:    " + str(I_recall) + "\nf1:        " +
     str(I_f1))

print("====================================================================================")

print("Considering about stop words, the score of multinomialNB,\naccuracy:  " + str(s_accuracy) +
     "\nprecision: " + str(s_precision) + "\nrecall:    " + str(s_recall) + "\nf1:        " +
     str(s_f1))
print("\n\n\n====================================================================================")
print("===============================using tf-idf=========================================")
print("====================================================================================")

print("Using tf-idf for words extraction, the score of multinomialNB,\naccuracy:  " + str(tfidf0_accuracy) +
     "\nprecision: " + str(tfidf0_precision) + "\nrecall:    " + str(tfidf0_recall) + "\nf1:        " +
     str(tfidf0_f1))

print("====================================================================================")

print("Using tf-idf for words extraction(with increased data), the score of multinomialNB,\naccuracy:  " + str(tfidf_accuracy) +
     "\nprecision: " + str(tfidf_precision) + "\nrecall:    " + str(tfidf_recall) + "\nf1:        " +
     str(tfidf_f1))

print("\n\nClassification Report for multinomialNB(bag-of words, increased data) after Resampling Dataset:\n")
print(classification_report(I_y_valid, predicted_MultinomialNB_2))

clf_SVC = SVC()
model_SVC = clf_SVC.fit(I_X, I_y)



# Score for k-fold validation using random-forest
rf_accuracy, rf_precision, rf_recall, rf_f1 = Model_Score(I_X, I_y, RandomForestClassifier(max_depth = 5, n_jobs = 2), 10)

print("====================================================================================")

print("Considering about stop words, the score of multinomialNB,\naccuracy:  " + str(rf_accuracy) +
    "\nprecision: " + str(rf_precision) + "\nrecall:    " + str(rf_recall) + "\nf1:        " +
    str(rf_f1))

# Builing random-forest Model (without any tuning or processing)
clf_rf = RandomForestClassifier(n_jobs = 2)
model_rf = clf_rf.fit(I_X_train, I_y_train)


bag_of_words_test = count.transform(raw_testing["article_words"])

# Create feature matrix
X_test = bag_of_words_test
y_test = raw_testing["topic"]

# Final test for multinomialNB
predicted_testcase = model_MultinomialNB_2.predict(X_test)
print("test case result(using the best performanced multinomailNB2):")
print("acccuracy: " + str(accuracy_score(y_test, predicted_testcase)))
print("precision: " + str(precision_score(y_test, predicted_testcase, average='macro')))
print("recall: " + str(recall_score(y_test, predicted_testcase, average='macro')))
print("f1: " + str(f1_score(y_test, predicted_testcase, average='macro')))
print(classification_report(y_test, predicted_testcase))
print("\n======================================================\n")

# Final test for SVC
predicted_SVC = model_SVC.predict(X_test)
print("test case result(using SVC):")
print("acccuracy: " + str(accuracy_score(y_test, predicted_SVC)))
print("precision: " + str(precision_score(y_test, predicted_SVC, average='macro')))
print("recall: " + str(recall_score(y_test, predicted_SVC, average='macro')))
print("f1: " + str(f1_score(y_test, predicted_SVC, average='macro')))
print(classification_report(y_test, predicted_SVC))
print("\n======================================================\n")

# Final test for random-forest
predicted_rf = model_SVC.predict(X_test)
print("test case result(using random-forest):")
print("acccuracy: " + str(accuracy_score(y_test, predicted_rf)))
print("precision: " + str(precision_score(y_test, predicted_rf, average='macro')))
print("recall: " + str(recall_score(y_test, predicted_rf, average='macro')))
print("f1: " + str(f1_score(y_test, predicted_rf, average='macro')))
print(classification_report(y_test, predicted_rf))

import operator

def TopTen (Model, X_test, y_test):
    predict_y = Model.predict(X_test)    #get the prediction
    proba_y = Model.predict_proba(X_test)  #get the probablity of each class
    recommendation = dict()
    
    for c in Model.classes_:
        recommendation[c] = dict()
        
    for i in range(len(proba_y)):
        pred = predict_y[i]         # get the predict for this sample
        proba = np.max(proba_y[i])  # get how much probability for this prediction
        recommendation[pred][i + 9501] = proba  #save article number and probability into dict

    for key in recommendation:
        d = recommendation[key]
        sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1),reverse=True))
        if (key == "IRRELEVANT"):   # skip irrelevant class
            continue
        print(key)
        count = 0
        for k in sorted_d:
            print("[" + str(count+1)+ "]  " + str(k) + "  " + str(sorted_d[k]))
            count += 1
            if count == 10:
                break


# get the topten recommendation using multinomialNB
TopTen(model_MultinomialNB_2, X_test, y_test)