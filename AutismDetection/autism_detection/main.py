"""
This is main.py contains the client code of the whole project.
| AUTHOR |    UPDATE    |   EMAIL                                 |
| LiuQM  |  2020/10/06  | contact:liuqm19@mails.tsinghua.edu.cn   |

TODO:
    
BUG:

Tricky:

"""
#%%
import os

import pandas as pd

from utils import get_config
from classification import grid_search, train_naive_bayesian, save_model, load_model, predict
from feature_selection import read_features, significant_difference_reduction
from preprocess import EyeTrackingBatchData, get_all_features


def main():
    config_file = "D:/src/AutismDetection/AutismDetection/docs/plr_config.json"  # need changed by user

    exs_conf, output_conf = get_config(config_file)

    all_features = get_all_features(exs_conf=exs_conf)

    features, labels = read_features(all_features=all_features)

    features, feature_list = significant_difference_reduction(features, labels)

    scores = grid_search(features, feature_list)

    scores.to_csv(output_conf["best_features_combination"], index=False)

    # Scores's columns is ['P', 'R', 'F1', 'ACC', 'AUC',
    #                      'feat1', 'feat2', 'feat3', 'feat4']
    # So we want features name is in [5:]
    best_four_features_combination = scores.iloc[0, 5:].dropna().tolist()

    _, best_model = train_naive_bayesian(all_features, best_four_features_combination)

    save_model(best_model, output_conf["best_model"])

    best_model = load_model(output_conf["best_model"])

    pred = predict(sample=all_features.iloc[0, :], model=best_model)

    if pred == 1:
        print("Autism Spectrum Disorder High Risky!!!")
    elif pred == 0:
        print("Typical Development.")


if __name__ == '__main__':
    main()

# %%
