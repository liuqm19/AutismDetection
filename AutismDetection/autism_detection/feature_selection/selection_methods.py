"""
This is selection_method.py contains some basic function of feature_selection
and the main() execution code.
| AUTHOR |    UPDATE    |   EMAIL                                 |
| LiuQM  |  2020/10/06  | contact:liuqm19@mails.tsinghua.edu.cn   |

TODO:

BUG:

Tricky:

"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

import utils.utils as utils


def read_features(feature_path=None, all_features=None) -> tuple:
    """
    Read features from file or .

    Args:
        feature_path: feature file path
        all_features: [DataFrame], n-m like feature map
    Return:
          result_features: features for all subjects
          labels: labels for all subjects

    Raise:
        1.There must one None between feature_path and features.
        2.Path does not exist.
    """
    if feature_path is not None and all_features is not None:
        raise Exception("There must one None between feature_path and features.")

    if feature_path is not None and all_features is None:
        if not os.path.exists(feature_path):
            raise Exception("Data path: {} does not exist! Please check!".format(feature_path))

        all_features = pd.read_csv(feature_path)

    result_features = all_features.set_index('name')
    result_features.index.name = 'name'
    labels = result_features['label']
    del result_features['label']

    # count_plot
    # ax = sns.countplot(labels, label="Count")
    # plt.show()

    return result_features, labels


def significant_difference_analysis(features, labels) -> tuple:
    """
    Do T-test between two kinds of people.

    Args:
        features:
        labels:

    Return:
        result: [DataFrame], significant difference features' fold_change and p_value
        significant_difference_feature_list:
    Raise:

    """

    columns = features.columns
    asd = features[labels == 1]
    td = features[labels == 0]

    # get fold_change(the difference between the average)
    asd_mean = asd.mean(axis=0)
    td_mean = td.mean(axis=0)
    fold_change = asd_mean - td_mean

    # t-test
    p_value = pd.Series(stats.ttest_ind(asd, td)[1], index=columns)

    result = pd.DataFrame({"fold_change": fold_change, 'p_value': p_value})
    result.index.name = 'name'
    result['log(p_value)'] = -np.log10(result['p_value'])
    result['sig'] = 'normal'

    result.loc[(result.fold_change > 0) & (result.p_value < 0.05), 'sig'] = 'up'
    result.loc[(result.fold_change < 0) & (result.p_value < 0.05), 'sig'] = 'down'

    ax = sns.scatterplot(x='fold_change', y='log(p_value)',
                         hue='sig',
                         hue_order=('down', 'normal', 'up'),
                         palette=("#377EB8", "grey", "#E41A1C"),
                         data=result)
    ax.set_xlabel('fold-change')
    ax.set_ylabel('-log(p_value)')
    plt.title("Volcano Map of Features")
    plt.show()

    result = result[result['p_value'] < 0.05]
    significant_difference_feature_list = result.index.values

    print("Significant features' information:")
    print(result)
    print("\n")
    return result, significant_difference_feature_list


def significant_difference_reduction(features, labels) -> tuple:
    """
    Do significant difference dimension reduction.

    Args:
        features:
        labels:

    Return:
        result_data: features after dimension reduction with significant difference reduction, and labels
        significant_difference_feature_list:
    Raise:

    """
    _, significant_difference_feature_list = significant_difference_analysis(features, labels)

    result_data = pd.DataFrame(features[significant_difference_feature_list],
                               columns=significant_difference_feature_list)

    result_data['label'] = labels.values

    return result_data, significant_difference_feature_list


def correlation_analysis(features, columns) -> pd.DataFrame:
    """
    Do correlation analysis with Person Correlation.

    Args:
        features: all features
        columns: significant_difference_feature_list

    Return:
        corr: high correlation features

    Raise:

    """
    features_to_corr = features[columns]
    features_to_corr.columns = [i+1 for i in range(columns.size)]
    features_to_corr.columns.name = 'Feature No.'

    corr = features_to_corr.corr()
    corr = corr.abs()
    # TODO:
    #     (LiuQM)Split corr operation and plot operation.

    mask = np.zeros_like(corr).astype(np.bool)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        sns.heatmap(corr, vmax=1, vmin=0,
                    cmap="YlGnBu",
                    mask=mask, annot=True)
        plt.title("Correlation Coefficient Matrix of Significant Difference Features")
        plt.show()

    corr = corr.where(mask)
    corr.index.name = None
    corr.columns.name = None
    corr = corr.stack().reset_index()
    corr.columns = ['feat1', 'feat2', 'corr']

    # Set the threshold value for high correlation.
    corr = corr[abs(corr['corr']) >= 0.75]

    # Drop duplicate.
    corr = corr[corr['feat1'] != corr['feat2']]

    corr['abs(corr)'] = abs(corr['corr'])
    corr = corr.sort_values('abs(corr)', ascending=False).reset_index(drop=True)

    print("Correlation analysis result:")
    print(corr)
    print("\n")

    return corr


def correlation_reduction(features, labels) -> pd.DataFrame:
    # TODO: Delete one of two features with high correlation and low log(p_value).
    return pd.DataFrame()


def pca_reduction(features, labels) -> pd.DataFrame:
    """
    Do PCA dimension reduction.

    Args:
        features:
        labels:

    Return:
        pca_data: features after dimension reduction with PCA, and labels

    Raise:

    """
    # Standardized
    data = StandardScaler().fit_transform(features)

    # Fitting a PCA converter for the feature.
    # Automatically select the number of principal components, must explain at least 95% of the original variance.
    pca_transformer = PCA(n_components=0.95)

    pca_data = pca_transformer.fit_transform(data)
    print(pca_transformer.explained_variance_ratio_)
    print("The first three principal components contain：", pca_transformer.explained_variance_ratio_[:3].sum())

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(labels)):
        ax.scatter(pca_data[i, 0], pca_data[i, 1], pca_data[i, 2],
                   marker=r'${}$'.format(labels[i]), s=60)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_zlabel("Principal component 3")
    plt.show()

    columns = []
    for i in range(pca_data.shape[1]):
        columns.append('pc'+str(i))

    pca_data = pd.DataFrame(pca_data, columns=columns)
    pca_data['label'] = labels.values

    return pca_data


def main():
    config_file = "D:/src/AutismDetection/AutismDetection/docs/plr_config.json"  # used changed by user

    _, output_conf = utils.get_config(config_file)

    features, labels = read_features(feature_path=output_conf['all_features'], all_features=None)

    significant_difference_features_info, significant_difference_feature_list \
        = significant_difference_analysis(features, labels)

    significant_difference_features_info.to_csv(output_conf['significant_difference_features_info'])

    corr = correlation_analysis(features, significant_difference_feature_list)

    significant_difference_features, significant_difference_feature_list = significant_difference_reduction(features,
                                                                                                            labels)


if __name__ == '__main__':
    main()

