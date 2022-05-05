import os
from os import walk
from typing import List

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
import seaborn as sea
import matplotlib.pyplot as plt

from utils import confusion_matrix, display_confusion_matrix, \
    display_classification_report_table, ClassificationReport


def prepare_data(file_name: str):
    df = pd.read_csv(file_name)
    df_columns = df.columns.values.tolist()

    features = df_columns[0:14]
    label = df_columns[14:]

    X = df[features]
    y = df[label]

    y = pd.get_dummies(y)  # one-hot

    # Question 2 / 3
    print(df['Class'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)


def part3():
    filenames = next(walk("data/predictions"), (None, None, []))[2]

    if not os.path.exists("data/images"):
        os.makedirs("data/images")

    for filename in filenames:
        # ignore y_test
        if filename == "y_test.csv":
            continue
        formatted = filename.replace(".csv", "").replace("y_pred_", "") \
            .replace("_", " ")
        f_formatted = filename.replace(".csv", ".png")
        print("Processing %s" % filename)
        generate_plots(y_test_file="data/predictions/y_test.csv",
                       y_pred_file="data/predictions/%s" % filename,
                       tree="DT" in filename,
                       confusion_plot_title="Modèle %s" % formatted,
                       confusion_plot_file="data/images/%s" % f_formatted)


def generate_plots(y_test_file: str = "",
                   y_pred_file: str = "",
                   tree: bool = True,
                   confusion_plot_file: str = None,
                   confusion_plot_title: str = "",
                   confusion_plot_cmap: str = "Reds"):
    """
    Generates a confusion plot and a classification report
    :param y_test_file: Test file path
    :param y_pred_file: Prediction file path
    :param tree: If prediction file is a tree or a neuronal network
    :param confusion_plot_file: Confusion plot file
                                (if not None, plot is saved to path)
    :param confusion_plot_title: Confusion plot title
    :param confusion_plot_cmap: Confusion plot heatmap color
    """
    df_test = pd.read_csv(y_test_file)
    df_pred = pd.read_csv(y_pred_file)

    # ravel => list of lists of int TO list of int
    y_true = df_test.values.ravel().tolist()
    y_pred = df_pred.values.tolist()

    # if not a tree, then get predicted value from neuronal network
    if not tree:
        new_pred = []
        for pred in y_pred:
            new_pred.append(np.argmax(pred, axis=0))
        y_pred = new_pred

    # Generate confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    # Generate plot for confusion matrix
    display_confusion_matrix(matrix, ["0", "1", "2", "3"],
                             cmap=confusion_plot_cmap,
                             title=confusion_plot_title,
                             file_name=confusion_plot_file)

    # classification metrics
    report = ClassificationReport(matrix)

    report.print_values()
    report.print_metrics()


if __name__ == '__main__':
    prepare_data('data/synthetic.csv')

    part3()
