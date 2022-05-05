from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea
from scipy.special import softmax


def confusion_matrix(y_test, y_pred) -> List[List[int]]:
    """
    Generates a confusion matrix for y_test and y_pret datasets
    :param y_test: Expected values
    :param y_pred: Predicted values
    :return: Confusion matrix
    """
    # set(y_test) for unique values not work, use np.unique
    class_nb = len(set(y_test))
    # initialize 0 matrix
    matrix = np.zeros((class_nb, class_nb))
    for i in range(len(y_test)):
        matrix[y_test[i]][y_pred[i]] += 1
    return matrix.astype(int).tolist()


def display_confusion_matrix(confusion_mtx: List,
                             class_names: List, annotation_font_size: int = 30,
                             label_font_size: int = 2, cmap: str = "Reds",
                             title: str = "", file_name: str = None):
    """
    Generates a plot using seaborn and matplotlib
    :param confusion_mtx: Confusion matrix (List of Lists)
    :param class_names: All class names
    :param annotation_font_size: Values font size
    :param label_font_size: Labels font size
    :param cmap: Heatmap colors
    :param title: Show's title on plot
    :param file_name: If file_name is not None : plot saved in file, else
                    plot is shown directly.
    """
    plt.figure(figsize=(8, 8))
    sea.set(font_scale=label_font_size)
    ax = sea.heatmap(
        confusion_mtx, annot=True,
        annot_kws={"size": annotation_font_size},
        cbar=False, cmap=cmap, fmt='d',
        xticklabels=class_names, yticklabels=class_names
    )
    ax.set(title=title, xlabel='Predicted label', ylabel='True label')

    if file_name is None:
        plt.show()
        return

    # Save figure
    plt.savefig(file_name)


def display_classification_report_table(data: List[List[float]],
                                        title: str = "Title",
                                        file_name: str = None):
    """
    Show's a plot with a table
    No fitting perfectly, so we didnt use it :(
    :param data: Matrix of data to add into table
    :param title:Show's title on plot
    :param file_name: If file_name is not None : plot saved in file, else
                plot is shown directly.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    column_labels = ["c1", "c2", "c3", "c4"]
    row_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    df = pd.DataFrame(data, columns=column_labels)
    ax.axis('tight')
    ax.axis('off')
    ax.set(title=title)

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     rowLabels=row_labels, loc="center", )

    table.set_fontsize(14)

    if file_name is None:
        plt.show(aspect='auto')
        return

    # Save figure
    plt.savefig(file_name)


def divide(a, b) -> float:
    """
    Division with 0 check
    :param a: numerator
    :param b: denominator
    :return: 0 if b == 0 else a/b
    """
    return a / b if b else 0


class ClassificationReport:
    FP: List[int]
    FN: List[int]
    TP: List[int]
    TN: List[int]
    class_count: int

    def __init__(self, matrix: List[List[int]]):
        """
        :param matrix: Confusion matrix
        """
        self.FP = []
        self.FN = []
        self.TP = []
        self.TN = []
        self.class_count = len(matrix)

        self.__process_confusion_matrix(matrix)

    def __process_confusion_matrix(self, matrix: List[List[int]]):
        """
        Calculates TP, FN, FP, TN for each class
        :param matrix: Confusion matrix
        """
        for i in range(self.class_count):
            self.TP.append(matrix[i][i])
            self.FN.append(sum(matrix[i]) - self.TP[i])
            self.FP.append(0)
            for j in range(self.class_count):
                self.FP[i] += matrix[j][i]
            self.FP[i] -= self.TP[i]
            self.TN.append(sum([sum(i) for i in zip(*matrix)]) - (
                    self.TP[i] + self.FN[i] + self.FP[i]))

    def sum_class_values(self, i: int = 0) -> int:
        """
        :param i: Class
        :return: Sum of TP, FN, TN, FP
        """
        return self.TP[i] + self.FN[i] + self.TN[i] + self.FP[i]

    def calculate_accuracy(self) -> List[float]:
        return [(divide(self.TP[i] + self.TN[i], self.sum_class_values(i))) for
                i in range(self.class_count)]

    def calculate_precision(self) -> List[float]:
        return [divide(self.TP[i], self.TP[i] + self.FP[i]) for i in
                range(self.class_count)]

    def calculate_recall(self) -> List[float]:
        return [divide(self.TP[i], self.TP[i] + self.FN[i]) for i in
                range(self.class_count)]

    def calculate_f1score(self) -> List[float]:
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        result = []
        for i in range(self.class_count):
            calc = divide(precision[i] * recall[i], precision[i] + recall[i])
            result.append(2 * calc)
        return result

    def print_metrics(self):
        titles = ["Accuracy", "Precision", "Recall", "F1-score"]
        row_format = "{:>15}" * (len(titles) + 1)
        print(row_format.format("", *titles))
        accuracy = self.calculate_accuracy()
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1score = self.calculate_f1score()

        for i in range(self.class_count):
            row = [
                i,
                "{:.2f}".format(accuracy[i]),
                "{:.2f}".format(precision[i]),
                "{:.2f}".format(recall[i]),
                "{:.2f}".format(f1score[i])
            ]
            print(row_format.format(*row))

        print()

    def print_values(self):
        titles = ["TP", "FP", "TN", "FN"]
        row_format = "{:>15}" * (len(titles) + 1)
        print(row_format.format("", *titles))

        for i in range(self.class_count):
            row = [i, self.TP[i], self.FP[i], self.TN[i], self.FN[i]]
            print(row_format.format(*row))

        print()

class Utility:
    @staticmethod
    def identity(Z):
        return Z, 1

    @staticmethod
    def tanh(Z):
        """
        Z : non activated outputs
        Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
        """
        A = np.empty(Z.shape)
        A = 2.0 / (1 + np.exp(-2.0 * Z)) - 1  # A = np.tanh(Z)
        df = 1 - A ** 2
        return A, df

    @staticmethod
    def sigmoid(Z):
        A = np.empty(Z.shape)
        A = 1.0 / (1 + np.exp(-Z))
        df = A * (1 - A)
        return A, df

    @staticmethod
    def relu(Z):
        A = np.empty(Z.shape)
        A = np.maximum(0, Z)
        df = (Z > 0).astype(int)
        return A, df

    @staticmethod
    def softmax(Z):
        return softmax(Z, axis=0)  # from scipy.special

    @staticmethod
    def cross_entropy_cost(y_hat, y):
        n = y_hat.shape[1]
        ce = -np.sum(y * np.log(y_hat + 1e-9)) / n
        return ce

    """
    Explication graphique du MSE:
    https://towardsdatascience.com/coding-deep-learning-for-beginners-linear-regression-part-2-cost-function-49545303d29f
    """

    @staticmethod
    def MSE_cost(y_hat, y):
        mse = np.square(np.subtract(y_hat, y)).mean()
        return mse

