import argparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression #, RandomizedLogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GroupKFold
import warnings
    
import pandas as pd
import numpy as np
from numpy import linalg
import random
import json
from collections import Counter
import os

# for analysis of features
import nltk
from scipy import stats

from myutils import Featurizer, get_size_tuple, PREFIX_WORD_NGRAM, PREFIX_CHAR_NGRAM

# fix seed for replicability
#seed=103
#seed=104
seed=105
random.seed(seed)
np.random.seed(seed)

def main():

    # parse command line options
    parser = argparse.ArgumentParser(description="""Classifier""")
    parser.add_argument("data", help="csv data file") # cross-validation if no --test file given
    parser.add_argument("--test", help="if given train on all and test on test file (otherwise CV)", required=False)
    parser.add_argument('--meta', help="provide path to alternative label file (gender|age file)")

    parser.add_argument("--folds", help="number of folds (if no test data is given)", default=5, type=int)
    parser.add_argument("--C", help="parameter C for regularization", required=False, default=1, type=float)
    parser.add_argument("--n-gram", help="word n-gram size, string separated by -", default="1-2")
    parser.add_argument("--c-n-gram", help="character n-gram size, string separated by -", default="0") # 0=off
    parser.add_argument("--show-instance", help="print instance", default=False, action="store_true")
    parser.add_argument("--analyze", help="analyze features", default=False, action="store_true")
    parser.add_argument("--classifier", default="SVM", choices=("LogReg","SVM"))
    parser.add_argument("--num-users", help="select up to k users [0=use all]", default=0, type=int)
    parser.add_argument("--embeds", help="embeddings file", default=False, action="store_true")
    parser.add_argument("--only-mean", help="mean embedding as only embed feature", default=False, action="store_true")

    parser.add_argument("--output", help="output predictions", required=False)


    args = parser.parse_args()

    if args.classifier == "LogReg":
        classifier = LogisticRegression()
    elif args.classifier == "SVM":
        classifier = LinearSVC(C=args.C)

    # print(classifier.classes_)
    print(classifier)

    user2meta=None
    if args.meta:
        # load labels
        user2meta = json.load(open(args.meta))

    emb = None
    if args.embeds:
        print("load embeddings..")
        emb = load_embeddings()

    ## read input data
    print("load data..")

    if args.test:

        X_train, y_train, X_test, y_test, vectorizer, group_ids = vectorize_data(args, test=args.test, embeds=emb, user2meta=user2meta)

        f1_test, acc_test = train_eval(args, X_train, y_train, X_test, y_test)
        print("weighted f1: {0:.1f}".format(f1_test * 100))
        print("accuracy: {0:.1f}".format(acc_test * 100))

        get_majority_baseline(y_train, y_test)

    else:
        ## if no --test file is given performe stratified CV

        X_all, y_all, vectorizer, group_ids = vectorize_data(args, embeds=emb, user2meta=user2meta)

        if args.meta:
            print("use group kfold")
            cv = GroupKFold(n_splits=args.folds)
            splitter = cv.split(X_all, y_all, group_ids)
        else:
            skf = StratifiedKFold(n_splits=args.folds)
            splitter = skf.split(X_all, y_all)
        f1_scores, acc_scores = [], []

        OUT=None
        if args.output:
            OUT = open(args.output,"w")
        fold_num=0
        for train, test in splitter:
            fold_num+=1
            print("========= fold {} ==========".format(fold_num))
            X_train, y_train = X_all[train], y_all[train]
            X_test, y_test = X_all[test], y_all[test]

            print("Labels in train data:", Counter(y_train))

            f1_test, acc_test = train_eval(classifier, X_train, y_train, X_test, y_test, output=OUT)

            f1_scores.append(f1_test)
            acc_scores.append(acc_test)
            print("weighted f1: {0:.1f}".format(f1_test * 100))

            if args.analyze:
                show_most_informative_features(classifier, vectorizer)

        if args.output:
            OUT.close()

        print("==================================")
        f1_scores = np.array(f1_scores)
        acc_scores = np.array(acc_scores)
        print("mean f1: {:.2f} ({:.2f})".format(np.mean(f1_scores) * 100, np.std(f1_scores) * 100))
        print("mean acc: {:.2f} ({:.2f})".format(np.mean(acc_scores) * 100, np.std(acc_scores) * 100))

        get_majority_baseline(y_train, y_test)

        print("size:", X_all.shape)

    # print out parameters
    for (a, v) in vars(args).items():
        print(a, v)


def train_eval(classifier, X_train, y_train, X_test, y_test, output=None):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        classifier.fit(X_train, y_train)

    y_predicted_test = classifier.predict(X_test)
    y_predicted_train = classifier.predict(X_train)

    if output:
        for pred, gold in zip(y_predicted_test, y_test):
            output.write(pred + "\t" + gold+ "\n")
    accuracy_dev = accuracy_score(y_test, y_predicted_test)
    accuracy_train = accuracy_score(y_train, y_predicted_train)
    print("Classifier accuracy train: {0:.2f}".format(accuracy_train*100))


    print("===== dev set ====")
    print("Classifier: {0:.2f}".format(accuracy_dev*100))

    #print(classification_report(y_test, y_predicted_test, digits=4))

    print(confusion_matrix(y_test, y_predicted_test, labels=classifier.classes_))
    return f1_score(y_test, y_predicted_test, average="weighted"), accuracy_score(y_test, y_predicted_test)

def get_majority_baseline(y_train, y_test):
    print("===")
    majority_label = Counter(y_train).most_common()[0][0]
    maj = [majority_label for x in range(len(y_test))]

    print("first instance")
    f1_maj, acc_maj = f1_score(y_test, maj, average="weighted"), accuracy_score(y_test, maj)
    print("Majority weighted F1: {0:.2f} acc: {1:.2f}".format(f1_maj * 100, acc_maj * 100))


def vectorize_data(args, test=None, embeds=None, user2meta=None):
    """
    :param args:
    :param X_train:
    :param X_test:
    :param get_mapping_org_transformed: True: keeps mapping original to transformed feature
    :return:
    """
    print("vectorize data..")

    # using pandas dataframe
    df_data = pd.read_csv(args.data)

    # use subset of users
    if args.num_users > 0:
        user_ids = list(df_data["user"].unique())
        print(user_ids)
        random.seed(seed)
        np.random.seed(seed)
        random.shuffle(user_ids)
        selected_users = user_ids[:args.num_users]
        print("selected users:", selected_users)
        df_data = df_data[df_data.user.isin(list(selected_users))]


    ## get label
    group=[]
    y_train = np.array(df_data["user"])
    if args.meta:
        group = y_train.copy() # keep user_ids for grouping
        y_train = np.array([user2meta[user] for user in y_train])

    print("number of users/classes:", len(set(y_train)))

    ### get data as dictionary

    X_train_dict = get_data_dictionary(args, df_data, embeds)
    print(X_train_dict[0])
    if test:
        X_test_dict = get_data_dictionary(args, df_data_test, embeds)
        y_test = np.array(df_data_test["user"])
        if args.meta:
            group = y_test.copy()
            y_test = np.array([user2meta[user] for user in y_test])
        assert(len(X_test_dict)==len(y_test))

    assert(len(X_train_dict)==len(y_train))

    dictVectorizer = DictVectorizer()

    X_train = dictVectorizer.fit_transform(X_train_dict)
    print("Vocab size:", len(dictVectorizer.vocabulary_))

    if test:
        X_test = dictVectorizer.transform(X_test_dict)

    if args.show_instance:
        print("first instance")
        print(X_train_dict[0])

    print(X_train.shape, len(y_train))

    if test:
        return X_train, y_train, X_test, y_test, dictVectorizer, group
    else:
        return X_train, y_train, dictVectorizer, group

def get_data_dictionary(args, data_frame, emb):

    data = [] # return list of dictionaries

    for index, row in data_frame.iterrows():
        d = {}
        for feat_name in row.keys():
            if feat_name in ["user", "session"]: # do not use!
                continue
            if feat_name != "text":
                d[feat_name] = float(row[feat_name])
            else:
                if not args.embeds:
                    f = Featurizer(word_ngrams=args.n_gram, char_ngrams=args.c_n_gram, binary=True)
                    di = f._ngrams(row["text"])
                    for k in di:
                        d[k] = di[k]
                else:
                    words = row["text"].split(" ")  # trivial tokenization
                    word_vec = [emb.get(w, emb["_UNK"]) for w in words]
                    avg_emb = np.mean(word_vec, axis=0)
                    sd_emb = np.std(word_vec, axis=0)
                    sum_emb = np.mean(word_vec, axis=0)

                    if args.only_mean:
                        for i, val in enumerate(avg_emb):
                                d["d_{}_{}".format(i, "mean")] = val
                    else:
                        for f, vec in (("mean", avg_emb), ("std", sd_emb), ("sum", sum_emb)):
                            for i, val in enumerate(vec):
                                d["d_{}_{}".format(i, f)] = val

                        d["overall_max"] = np.max(word_vec)
                        d["overall_min"] = np.min(word_vec)
                        d["emb_cov_rate"] = np.sum([1 for w in words if w in emb])/len(words)

        data.append(d)

    return data


def show_most_informative_features(clf, vectorizer, n=10):
    feature_names = vectorizer.get_feature_names()
    for i in range(0,len(clf.coef_)):
        coefs_with_fns = sorted(zip(clf.coef_[i], feature_names))
        top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
        print("i",i, clf.classes_[i])
        for (coef_1, fn_1), (coef_2, fn_2) in top:
            print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

def load_embeddings():
    file_name= "embeds/poly_a/en.polyglot.txt"
    emb = {}
    for line in open(file_name):
        fields = line.split()
        vec = [float(x) for x in fields[1:]]
        word = fields[0]
        emb[word] = vec
        # emb[word] /= linalg.norm(emb[word])
    return emb


if __name__=="__main__":
    main()

