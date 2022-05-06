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

from neuronalnetwork import NeuralNet
from utils import confusion_matrix, display_confusion_matrix, \
    display_classification_report_table, ClassificationReport, Utility


def part2(file_name: str):
    df = pd.read_csv(file_name)
    df_columns = df.columns.values.tolist()

    features = df_columns[0:14]
    label = df_columns[14:]

    # Normalize data
    for col in features:
        big = np.max(df[col])
        df[col] /= big

    X = df[features]

    y = pd.get_dummies(df, columns=label)  # one-hot
    # One-hot not working wtf?
    y = y[y.columns.values.tolist()[14:]]  # fix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)

    print(y_test)

    # Initialization of the model
    model = NeuralNet(X_train=X_train, y_train=y_train, X_test=X_test,
                      y_test=y_test, activation='tanh', epoch=100,
                      hidden_layer_sizes=(10, 8, 6), heuristic='he-et-al')
    # Fitting the model with data
    gr_train, gr_test = model.train(X_train, y_train, X_test, y_test)

    accuracies, y_pred = model.predict(X_test, y_test)
    print("Accuracy: " + str(np.mean(accuracies)))
    print(y_pred)
    print(y_test)

    new_pred = []
    for pred in y_pred:
        new_pred.append(np.argmax(pred, axis=0))
    y_pred = new_pred

    print(y_pred)

    y_test = np.argmax(y_test.values, axis=1)

    print(y_test)

    # Confusion matrix
    matrix = confusion_matrix(y_pred=y_pred, y_test=y_test)
    print(matrix)

    # Plotting the graph of the errors
    plt.plot(gr_train, label='train')
    plt.plot(gr_test, label='test')
    plt.legend()
    plt.title('NN relu 6-4')
    plt.show()

    display_confusion_matrix(matrix, class_names=["0", "1", "2", "3"])



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
    part2('data/synthetic.csv')

    # part3()
