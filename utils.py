from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea

from scipy.special import \
    softmax  # use built-in function to avoid numerical instability


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
    def softmax_gradient(z):
        """
        Source: https://github.com/eliben/deep-learning-samples/blob/master/softmax/softmax.py

        Computes the gradient of the softmax function.
        z: (T, 1) array of input values where the gradient is computed. T is the
           number of output classes.
        Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
        is DjSi - the partial derivative of Si w.r.t. input j.
        """
        Sz = softmax(z)
        # -SjSi can be computed using an outer product between Sz and itself. Then
        # we add back Si for the i=j cases by adding a diagonal matrix with the
        # values of Si on its diagonal.
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D

    @staticmethod
    def cross_entropy_loss(p, y):
        """
        Source: https://github.com/eliben/deep-learning-samples/blob/master/softmax/softmax.py

        Cross-entropy loss between predicted and expected probabilities.
        p: vector of predicted probabilities.
        y: vector of expected probabilities. Has to be the same shape as p.
        Returns a scalar.
        """
        assert (p.shape == y.shape)
        return -np.sum(y * np.log(p))

    @staticmethod
    def cross_entropy_cost(y_hat, y):
        n = y_hat.shape[1]
        ce = -np.sum(y * np.log(y_hat + 1e-9)) / n
        return ce

    @staticmethod
    def cross_entropy(y, y_pre):
        loss = -np.sum(y * np.log(y_pre))
        return loss / float(y_pre.shape[0])

    """
    Explication graphique du MSE:
    https://towardsdatascience.com/coding-deep-learning-for-beginners-linear-regression-part-2-cost-function-49545303d29f
    """

    @staticmethod
    def MSE_cost(y_hat, y):
        mse = np.square(np.subtract(y_hat, y)).mean()
        return mse


def confusion_matrix(y_test, y_pred) -> List[List[int]]:
    """
    Generates a confusion matrix for y_test and y_pret datasets
    :param y_test: Expected values
    :param y_pred: Predicted values
    :return: Confusion matrix
    """
    # set(y_test) for unique values not work, use np.unique
    class_nb = len(np.unique(y_test))
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


def calculate_accuracy(y_pred, y_real):
    """
    Calculate accuracy
    :param y_pred: Predictions
    :param y_real: Real values (y_test)
    :return: Accuracy
    """
    max_y_pred = np.where(y_pred == np.amax(y_pred))[0][0]
    max_y_real = np.where(y_real == np.amax(y_real))[0][0]
    return int(max_y_pred == max_y_real)

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
