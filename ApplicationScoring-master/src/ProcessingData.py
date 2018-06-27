import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
#import spacy
import re
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

#embedding_model = spacy.load("en_core_web_md")

#Read the data from the files.
def read_application_data(app_file, score_file):
    app_data = open(app_file, "r")
    app_data_map = {}
    count = 0
    section_list = set()
    for line in app_data:
        count += 1
        if count == 1:
            continue

        elements = line.split(",")
        reference = elements[0]
        section = elements[3]
        section_list.add(section)
        # heading = elements[4]
        #input data, the features used.
        input = elements[5]
        if reference not in app_data_map:
            app_data_map[reference]= {}

        if section not in app_data_map[reference]:
            app_data_map[reference][section] = []

        if input is not None and len(input) > 0 and "NULL" not in input:
            app_data_map[reference][section].append(input)
    app_data.close()

    score_data = open(score_file, "r")
    score_data_map = {}
    count = 0
    for line in score_data:
        count += 1
        if count == 1 or "(Select)" in line:
            continue

        elements = line.split(",")
        reference = elements[0]
        score = elements[4]
        if reference not in score_data_map:
            score_data_map[reference]= []

        if score is not None and len(score) > 0:
            score_data_map[reference].append(int(score))
    score_data.close()

    return list(section_list), app_data_map, score_data_map


#extract the features from the app_data_map
def extract_features(app_data_map, section_list):
    all_application = []
    all_vectors = []

    for reference in app_data_map:
        an_application = app_data_map[reference]
        vectorlised_app = []
        text = ""
        for section in section_list:
            if section in an_application:
                length = len(an_application[section])
                if length > 0:
                    length = 1

                vectorlised_app.append(length)
                input = process_text(" ".join(an_application[section]))
                if len(input) > 0:
                    text = text + " " + input
            else:
                vectorlised_app.append(0)
        # print len(vectorlised_app)
        # vectorlised_app = []
        #embedding_vector = embedding_model(text)
        # vectorlised_app.extend(embedding_vector.vector)

        all_vectors.append(vectorlised_app)
        all_application.append(text)

    tfidf_vectoriser = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, \
                                       max_features=500, norm='l2')
    counts = tfidf_vectoriser.fit_transform(all_application)
    n_grams_features = counts.toarray()
    features = np.append(all_vectors, n_grams_features, 1)
    print(len(features[0]))
    minmaxsc = MinMaxScaler()
    features =  minmaxsc.fit_transform(features)
    return features


def process_text(text):
    if type(text) != str:
        text = np.unicode(text).encode('utf8').strip()
    if len(text.strip()) == 0 or "(Select)" in text:
        return ""

    try:
        text = text.decode('utf-8')
    except:
        text = re.sub("\\W", " ", text).strip()

    text = re.sub("<.*?>|&.*?;", " ", text)
    tokenized_text = nltk.word_tokenize(text)
    processed_text = u' '.join(tokenized_text)
    return processed_text


def process_score_data(score_data_map):
    average_score_map = {}
    for reference in score_data_map:
        if len(score_data_map[reference]) > 0:
            average = np.average(np.asanyarray(score_data_map[reference]))
            average_score_map[reference] = average
    return average_score_map


def run(app_file, score_file):
    section_list, app_data_map, score_data_map = read_application_data(app_file, score_file)

    score_data = process_score_data(score_data_map)
    data = {}
    scores = []
    references = []
    for reference in score_data:
        scores.append(score_data[reference])
        data[reference] = app_data_map[reference]
        references.append(reference)

    features = extract_features(data, section_list)
    print(features.shape)

    clf = RandomForestRegressor(n_estimators=100)
    train_data, valid_data = split_kfolds(features, scores, references, 5)
    rmse_scores = []
    for i in range(len(train_data)):
        train_features = train_data[i]["features"]
        train_scores = train_data[i]["scores"]
        train_references = train_data[i]["references"]

        valid_features = valid_data[i]["features"]
        valid_scores = valid_data[i]["scores"]
        valid_references = valid_data[i]["references"]

        sel = VarianceThreshold(threshold=(.005 * (1 - .2)))
        train_features = sel.fit_transform(train_features)
        valid_features = sel.transform(valid_features)


        feature_selector = SelectKBest(mutual_info_regression, k=325)
        train_features = feature_selector.fit_transform(train_features, train_scores)
        valid_features = feature_selector.transform(valid_features)

        clf.fit(train_features, train_scores)
        valid_predicted_scores = clf.predict(valid_features)
        train_predicted_scores = clf.predict(train_features)
        rmse_on_train = mean_squared_error(train_predicted_scores, train_scores) ** 0.5
        rmse_on_valid = mean_squared_error(valid_predicted_scores, valid_scores) ** 0.5

        print("Model Test RMSE: %f" % rmse_on_valid)
        print("Model Train RMSE: %f" % rmse_on_train)
        print(train_features.shape)
        print(valid_features.shape)
        for j in range(len(valid_references)):
            print (valid_references[j], valid_scores[j], valid_predicted_scores[j])

        rmse_scores.append(rmse_on_valid)

    rmse_scores = np.asarray(rmse_scores)

    return rmse_scores


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def split_kfolds(features, scores, references, n_fold):
    kf = KFold(n_splits=n_fold)
    train_data =[]
    valid_data = []
    features = np.asarray(features)
    scores = np.asarray(scores)



    references = np.asarray(references)
    for train_index, valid_index in kf.split(features):
        train = {"features": features[train_index], "scores": scores[train_index], "references": references[train_index]}
        valid = {"features": features[valid_index], "scores": scores[valid_index], "references": references[valid_index]}
        train_data.append(train)
        valid_data.append(valid)
    return train_data, valid_data


if __name__ == "__main__":
    scores = run("../dataset/RIRApplicationForms_12-03-2018.csv",
                          "../dataset/scores.csv")
    display_scores(scores)





