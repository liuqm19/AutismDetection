"""
This is algorithms.py contains some basic function of classification two types of data
and the main() execution code.
| AUTHOR |    UPDATE    |   EMAIL                                 |
| LiuQM  |  2020/10/06  | contact:liuqm19@mails.tsinghua.edu.cn   |

TODO:
    
BUG:

Tricky:
    Used grid_search and naive_bayesian to find out best 4 features combination
    under the circumstance of two-types classification problem.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import utils.utils as utils


def evaluation(y_true, y_pred_prob, model_choice='NB') -> list:
    """
    Evaluate the performance of a model by real and predictive label of model.

    Args:
        y_true: the real label.
        y_pred_prob: the predictive label.
        model_choice: The kind of classification model we used.
    Return:
        [p, r, f1, acc, auc]: precision, recall, f1, accuracy, auc

    Raise:

    """
    y_pred = np.argmax(y_pred_prob, axis=1)

    # uncomment it if you want a confusion matrix plot.
    # plot_confmat(y_true, y_pred, model_choice)

    p = precision_score(y_true=y_true, y_pred=y_pred)
    r = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    y_pos_prob = y_pred_prob[:, 1]  # estimated probability of positive class
    auc = roc_auc_score(y_true, y_pos_prob)

    return [p, r, f1, acc, auc]


def plot_confmat(y_true, y_pred, model_choice='NB') -> None:
    """
    Draw the confusion matrix.

    Args:
        y_true: [list-like], the real label.
        y_pred: [list-like], the predictive label.
        model_choice: The kind of classification model we used.

    Return:

    Raise:

    """
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title(model_choice)
    plt.show()


def grid_search_5f(data, columns) -> pd.DataFrame:
    """
    Uses grid search method to find out the best 5 features combination.

    Args:
        data: n - m like features map, n is the number of subjects and m is the number of features(plus name and label).
        columns: the features used to search.

    Return:
        scores: [DataFrame], contains all 5 features combination and their evaluation metrics.

    Raise:

    """
    scores_list = []
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            for k in range(j+1, len(columns)):
                for m in range(k+1, len(columns)):
                    for n in range(m+1, len(columns)):
                        feat_list = [columns[i], columns[j],
                                     columns[k], columns[m],
                                     columns[n]]
                        scores, _ = train_naive_bayesian(data, feat_list)
                        scores_list.append(scores)

    scores = pd.DataFrame(data=scores_list,
                          columns=['P', 'R', 'F1', 'ACC', 'AUC',
                                   'feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    scores = scores.sort_values(["F1", 'ACC', "AUC"], ascending=False).reset_index(drop=True)
    return scores.head()


def grid_search_4f(data, columns) -> pd.DataFrame:
    """
    Uses grid search method to find out the best 4 features combination.

    Args:
        data: n - m like features map, n is the number of subjects and m is the number of features(plus name and label).
        columns: the features used to search.

    Return:
        scores: [DataFrame], contains all 4 features combination and their evaluation metrics.

    Raise:

    """
    scores_list = []
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            for k in range(j+1, len(columns)):
                for m in range(k+1, len(columns)):
                    feat_list = [columns[i], columns[j],
                                 columns[k], columns[m]]
                    scores, _ = train_naive_bayesian(data, feat_list)
                    scores_list.append(scores)

    scores = pd.DataFrame(data=scores_list,
                          columns=['P', 'R', 'F1', 'ACC', 'AUC',
                                   'feat1', 'feat2', 'feat3', 'feat4'])
    scores = scores.sort_values(["F1", 'ACC', "AUC"], ascending=False).reset_index(drop=True)
    return scores.head()


def grid_search_3f(data, columns) -> pd.DataFrame:
    """
    Uses grid search method to find out the best 3 features combination.

    Args:
        data: n - m like features map, n is the number of subjects and m is the number of features(plus name and label).
        columns: the features used to search.

    Return:
        scores: [DataFrame], contains all 3 features combination and their evaluation metrics.

    Raise:

    """
    scores_list = []
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            for k in range(j+1, len(columns)):
                feat_list = [columns[i], columns[j],
                             columns[k]]
                scores, _ = train_naive_bayesian(data, feat_list)
                scores_list.append(scores)

    scores = pd.DataFrame(data=scores_list,
                          columns=['P', 'R', 'F1', 'ACC', 'AUC',
                                   'feat1', 'feat2', 'feat3'])
    scores = scores.sort_values(["F1", 'ACC', "AUC"], ascending=False).reset_index(drop=True)
    return scores.head()


def grid_search_2f(data, columns) -> pd.DataFrame:
    """
    Uses grid search method to find out the best 2 features combination.

    Args:
        data: n - m like features map, n is the number of subjects and m is the number of features(plus name and label).
        columns: the features used to search.

    Return:
        scores: [DataFrame], contains all 2 features combination and their evaluation metrics.

    Raise:

    """
    scores_list = []
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            feat_list = [columns[i], columns[j]]
            scores, _ = train_naive_bayesian(data, feat_list)
            scores_list.append(scores)

    scores = pd.DataFrame(data=scores_list,
                          columns=['P', 'R', 'F1', 'ACC', 'AUC',
                                   'feat1', 'feat2'])
    scores = scores.sort_values(["F1", 'ACC', "AUC"], ascending=False).reset_index(drop=True)
    return scores.head()


def train_naive_bayesian(data, feat_list) -> tuple:
    """
    Train naive bayesian model and evaluate it.

    Args:
        data: n - m like features map, n is the number of subjects and m is the number of features.
        feat_list: the features used to train model.

    Return:
        scores: [list], 4 features combination and their evaluation metrics.
        model: [dict], contains a standardScaler and some models, using Thinking of Integrated Learning.

    Raise:

    """
    X = data[feat_list].values
    y = data['label'].values

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    leave_one_out = LeaveOneOut()
    leave_one_out.get_n_splits(X)

    label, y_pred = [], []

    models = []

    for train_index, test_index in leave_one_out.split(X):
        clf = GaussianNB()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_pred_prob = clf.predict_proba(X_test)

        label.append(y_test[0])
        y_pred.append(y_pred_prob[0])

        # Using Thinking of Integrated Learning
        models.append(clf)

    label = np.array(label)
    y_pred = np.array(y_pred)

    model = {"Feat_list": feat_list,
             "StandardScaler": scaler,
             "Models": models}

    scores = evaluation(label, y_pred)
    scores.extend(feat_list)

    print("\n")
    print("features combination: {}".format(feat_list))
    print('precision:%.3f' % scores[0])
    print('recall:%.3f' % scores[1])
    print('F1:%.3f' % scores[2])
    print('accuracy:%.3f' % scores[3])
    print('auc:%.3f' % scores[4])
    print("\n")

    return scores, model


def get_model(model_choice='NB'):
    """
    Train naive bayesian model and evaluate it.

    Args:
        model_choice: 'NB', 'LR', 'SVM', 'LDA', 'RF'

    Return:
        model_choice: to select which kind of classification model we would like to use.

    Raise:

    """
    def NB():  # first case
        print('This is the NB')
        return GaussianNB()

    def LR():  # second case
        print('This is the LR')
        return LogisticRegressionCV(cv=3)

    def SVM():  # third case
        print('This is the SVM')
        return SVC(kernel='linear', probability=True)

    def LDA():
        print('This is the LDA')
        return LinearDiscriminantAnalysis()

    def RF():
        print('This is the RF')
        return RandomForestClassifier(n_estimators=10)

    def default():
        NB()

    switch = {'NB': NB,  # Be careful not to add parentheses here
              'LR': LR,
              'SVM': SVM,
              'LDA': LDA,
              'RF': RF
              }
    # Execute the corresponding function, if not exist, execute the default function
    clf = switch.get(model_choice, default)()
    return clf


def train_model(data, feat_list, model_choice='NB') -> tuple:
    """
    Train classification model and evaluate it.

    Args:
        data: n - m like features map, n is the number of subjects and m is the number of features.
        feat_list: the features used to train model.
        model_choice: to select which kind of classification model we would like to use.

    Return:
        scores: [list], 4 features combination and their evaluation metrics.
        model: [dict], contains a standardScaler and some models, using Thinking of Integrated Learning.

    Raise:

    """
    X = data[feat_list].values
    y = data['label'].values

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    leave_one_out = LeaveOneOut()
    leave_one_out.get_n_splits(X)

    label, y_pred = [], []

    models = []

    for train_index, test_index in leave_one_out.split(X):
        clf = get_model(model_choice)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_pred_prob = clf.predict_proba(X_test)

        label.append(y_test[0])
        y_pred.append(y_pred_prob[0])

        # Using Thinking of Integrated Learning
        models.append(clf)

    label = np.array(label)
    y_pred = np.array(y_pred)

    model = {"Feat_list": feat_list,
             "StandardScaler": scaler,
             "Models": models}

    scores = evaluation(label, y_pred, model_choice)
    scores.extend([model_choice])
    print("\n")
    print("features combination: {}".format(feat_list))
    print('precision:%.3f' % scores[0])
    print('recall:%.3f' % scores[1])
    print('F1:%.3f' % scores[2])
    print('accuracy:%.3f' % scores[3])
    print('auc:%.3f' % scores[4])
    print("\n")

    return scores, model, y_pred


def save_model(model, path) -> None:
    """
    Save model.

    Args:
        model: what model to save
        path: where to save

    Return:

    Raise:
        1.Path does not exist.
        2.Model is None.
    """
    if not os.path.exists(os.path.dirname(path)):
        raise Exception("Path {} does not exist. Please check.".format(os.path.dirname(path)))
    if model is None:
        raise Exception("Model is None. Please check.")
    joblib.dump(model, path)


def load_model(path) -> any:
    """
    Load model.

    Args:
        path: where to load

    Return:

    Raise:
        1.Path does not exist.
    """
    if not os.path.exists(path):
        raise Exception("Path {} does not exist. Please check.".format(path))

    model = joblib.load(path)

    return model


def predict(sample, model) -> int:
    """
    Predict sample's label using naive bayesian model.

    Args:
        sample: [DataFrame], 1 - m like features map, n is the number of subjects and m is the number of features
        model: the model used to predict

    Return:
        y_pred: predicted label

    Raise:

    """
    features_combination = model["Feat_list"]
    scaler = model["StandardScaler"]
    models = model["Models"]

    features = sample[features_combination].values
    features = features.reshape((1, -1))
    features = scaler.transform(features.astype(float))

    y_pred = []
    for clf in models:
        y_pred_prob = clf.predict_proba(features)
        y_pred.append(y_pred_prob[0])

    y_pred = np.array(y_pred)
    y_pred = y_pred.mean(axis=0)
    y_pred = int(np.argmax(y_pred))

    return y_pred


def grid_search(data, columns) ->pd.DataFrame:
    """
    Use grid search method to find best features combination(2f-5f).

    Args:
        data:
        columns:

    Returns:
        scores: DataFrame, [P, R, F1, ACC, AUC]

    Raises:

    """
    scores_2f = grid_search_2f(data, columns)
    scores_3f = grid_search_3f(data, columns)
    scores_4f = grid_search_4f(data, columns)
    #scores_5f = grid_search_5f(data, columns)
    scores = pd.concat([scores_2f, scores_3f, scores_4f], axis=0)
    scores = scores.sort_values(["F1", "AUC", 'ACC'], ascending=False).reset_index(drop=True)
    return scores


def find_best_features_combination(features, features_name) ->tuple:
    """
    Use grid search method to find best features combination(2f-5f).

    Args:
        features:
        features_name:

    Returns:
        scores: DataFrame, [P, R, F1, ACC, AUC]
        best_features_combination: list, [feat1, feat2,...]

    Raises:

    """
    # Use grid search to find out the best features combination.
    scores = grid_search(features, features_name)

    # Scores's columns is ['P', 'R', 'F1', 'ACC', 'AUC',
    #                      'feat1', 'feat2', 'feat3', 'feat4', ...]
    # So we want features name is in [5:]
    best_features_combination = scores.iloc[0, 5:].dropna().tolist()

    return scores, best_features_combination


def find_best_model(data, best_features_combination):

    scores_NB, model_NB, _ = train_model(data, best_features_combination, model_choice='NB')
    scores_LR, model_LR, _ = train_model(data, best_features_combination, model_choice='LR')
    scores_SVM, model_SVM, _ = train_model(data, best_features_combination, model_choice='SVM')
    scores_LDA, model_LDA, _ = train_model(data, best_features_combination, model_choice='LDA')
    scores_RF, model_RF, _ = train_model(data, best_features_combination, model_choice='RF')

    scores = pd.DataFrame(data=[scores_NB, scores_LR, scores_SVM, scores_LDA, scores_RF],
                          columns=['P', 'R', 'F1', 'ACC', 'AUC', 'model_choice'])
    scores = scores.sort_values(["F1", 'ACC', "AUC"], ascending=False).reset_index(drop=True)
    models = {
        'NB': model_NB,
        'LR': model_LR,
        'SVM': model_SVM,
        'LDA': model_LDA,
        'RF': model_RF
    }
    best_scores = scores.iloc[0, :5]
    best_models = models[scores.iloc[0, 5]]
    return scores, best_scores, best_models


def train_fusion_model(plr_data, plr_feat_list, face_data, face_feat_list):
    """
    Train fusion classification model and evaluate it.

    Args:
        plr_data: n - m like features map, n is the number of subjects and m is the number of features.
        plr_feat_list: the name of features used to train model.
        face_data: n - m like features map, n is the number of subjects and m is the number of features.
        face_feat_list: the name of features used to train model.

    Return:
        weight_related_scores: [list], weights and their evaluation metrics.

    Raise:

    """
    merged_data = merge_experiments_data(plr_data, face_data)
    label = merged_data['label'].values

    _, _, plr_y_pred = train_model(merged_data, plr_feat_list, model_choice='LDA')
    _, _, face_y_pred = train_model(merged_data, face_feat_list, model_choice='NB')
    scores_list = []
    for weight in range(0, 101, 1):
        weight1 = 0.01*weight
        weight2 = (1-0.01*weight)
        y_pred = weight1*plr_y_pred + weight2*face_y_pred
        scores = evaluation(label, y_pred, model_choice='fusion')
        scores.extend([weight1, weight2])
        scores_list.append(scores)
    scores = pd.DataFrame(data=scores_list,
                          columns=['P', 'R', 'F1', 'ACC', 'AUC', 'weight1', 'weight2'])
    scores = scores.sort_values(['F1', 'AUC', 'ACC'], ascending=False).reset_index(drop=True)

    return scores


def prepare_best_features(config_file):
    """
    Parse config_file

    Args:
        config_file: file need to parse

    Returns:
        features:
        significant_difference_features_name:
        best_features_combination:
        output_conf:
    Raises:

    """
    exs_conf, output_conf = utils.get_config(config_file)

    # Read data and significant features data.
    features = pd.read_csv(output_conf['all_features'])
    significant_difference_features_info = pd.read_csv(output_conf['significant_difference_features_info'])
    significant_difference_features_name = significant_difference_features_info['name'].tolist()

    # Read best features combination
    best_features_combination = pd.read_csv(output_conf["best_features_combination"])
    best_features_combination = best_features_combination.iloc[0, 5:].dropna().tolist()

    return features, significant_difference_features_name, best_features_combination, output_conf


def merge_experiments_data(plr_data, face_data):
    """
    merge data

    Args:
        plr_data: data from pupil-to-light experiment
        face_data: data from upright-face-recognition experiment

    Returns:
        return_data: merged data to return
    Raises:

    """
    return_data = pd.merge(plr_data, face_data, on=['name', 'label'])
    return return_data


def fusion_main():
    plr_config_file = "D:/src/AutismDetection/AutismDetection/docs/plr_config.json"
    face_config_file = "D:/src/AutismDetection/AutismDetection/docs/face_config.json"
    plr_data, _, plr_best_features_combination, _ = prepare_best_features(plr_config_file)
    face_data, _, face_best_features_combination, _ = prepare_best_features(face_config_file)

    try:
        scores = train_fusion_model(plr_data, plr_best_features_combination, face_data, face_best_features_combination)
        scores.to_csv("G:/AutismDetection/csv/fusion_scores.csv", index=False)
    except Exception as e:
        print(e)


def single_model_main():
    config_file = "D:/src/AutismDetection/AutismDetection/docs/plr_config.json"

    exs_conf, output_conf = utils.get_config(config_file)

    # Read data and significant features data.
    features = pd.read_csv(output_conf['all_features'])
    significant_difference_features_info = pd.read_csv(output_conf['significant_difference_features_info'])
    significant_difference_features_name = significant_difference_features_info['name'].tolist()

    # Use grid search and naive bayes to find out the best features combination.
    scores, best_features_combination = find_best_features_combination(features, significant_difference_features_name)

    # Store the result scores.
    scores.to_csv(output_conf["best_features_combination"], index=False)

    scores, _, best_models = find_best_model(features, best_features_combination)

    scores.to_csv(output_conf["scores_models"], index=False)
    save_model(best_models, output_conf["best_model"])

    model = load_model(output_conf["best_model"])

    pred = predict(sample=features.iloc[0, :], model=model)

    if pred == 1:
        print("Autism Spectrum Disorder High Risky!!!")
    elif pred == 0:
        print("Typical Development.")


if __name__ == '__main__':
    single_model_main()
