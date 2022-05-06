from typing import List, Tuple

import numpy as np
from sklearn.utils import shuffle

from utils import Utility, calculate_accuracy


class NeuralNet:
    hidden_layer_sizes: Tuple
    layer_sizes: Tuple
    learning_rate: float
    epoch: int

    bias: List
    weights: List

    A: List
    df: List

    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None,
                 hidden_layer_sizes=(6, 4), activation='identity',
                 learning_rate=0.01, epoch=150, heuristic: str = 'xavier'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.layer_sizes = (len(X_train.columns),) + hidden_layer_sizes + (
            len(y_train.columns),)

        # Select activation function
        if activation == 'relu':
            self.activation = Utility.relu
        elif activation == 'tanh':
            self.activation = Utility.tanh
        elif activation == 'sigmoid':
            self.activation = Utility.sigmoid
        else:
            self.activation = Utility.identity

        # init weights and biases
        self.__initialization(heuristic)

    def __initialization(self, heuristic: str = 'xavier'):
        self.A = (len(self.hidden_layer_sizes) + 1) * [None]
        self.df = (len(self.hidden_layer_sizes) + 1) * [None]
        self.bias = []
        self.weights = []
        for i in range(1, len(self.layer_sizes)):
            # weights init using Xavier method
            self.weights.append(np.array(
                self.layer_sizes[i] * [self.layer_sizes[i - 1] * [None]]))

            # Xavier is default init method
            scale = 1.0 / np.sqrt(self.layer_sizes[i])

            if heuristic == 'he-et-al':
                scale = 2.0 / np.sqrt(self.layer_sizes[i])

            self.weights[i - 1] = np.random.normal(
                loc=0.0,
                scale=scale,
                size=(self.layer_sizes[i], self.layer_sizes[i - 1])
            )
            # initialization biases with zeros matrix
            self.bias.append(np.array(self.layer_sizes[i] * [None]))
            self.bias[i - 1] = np.zeros(self.layer_sizes[i])

    def prop_forward(self, X, y):
        # Z (ponderations)
        # W (weights)
        # A (activations)
        # b (biais)
        Z = len(self.layer_sizes) * [None]
        # First values are input values
        input_layer = X

        for layer in range(len(self.hidden_layer_sizes) + 1):
            # Z[l] = W[l] * A[l-1] + b[l]
            Z[layer + 1] = np.dot(self.weights[layer], input_layer) \
                           + self.bias[layer]

            # Ouput layer is calculated with softmax
            if layer == len(self.hidden_layer_sizes):
                self.A[layer] = Utility.softmax(Z[layer + 1])
                self.df[layer] = Utility.softmax_gradient(Z[layer + 1])
            else:
                # activation is the selected activation function
                # should return matrix and derivative
                self.A[layer], self.df[layer] = self.activation(Z[layer + 1])

            # Next input_layer is actual layer activations
            input_layer = self.A[layer]
        # Calculate error from output layer
        error = Utility.cross_entropy(y, self.A[-1], )

        return error, self.A[-1]

    def prop_backward(self, X, y):
        # Initialization
        delta = (len(self.hidden_layer_sizes) + 1) * [None]
        dW = (len(self.hidden_layer_sizes) + 1) * [None]
        db = (len(self.hidden_layer_sizes) + 1) * [None]
        # First calculate output layer
        delta[-1] = self.A[-1] - y
        dW[-1] = np.transpose(delta[-1] * self.A[-2][:, np.newaxis])
        db[-1] = delta[-1]

        # Then calculate all remaining layers (back to front)
        for layer in range(len(self.hidden_layer_sizes) - 1, -1, -1):
            delta[layer] = np.multiply(
                np.dot(self.weights[layer + 1].T, delta[layer + 1]),
                self.df[layer]
            )

            x_trans = X if layer == 0 else self.A[layer - 1]

            dW[layer] = np.transpose(
                delta[layer] * x_trans[:, np.newaxis]
            )
            db[layer] = delta[layer]

        # Update weights, and biases
        for layer in range(len(self.hidden_layer_sizes) + 1):
            self.weights[layer] = self.weights[layer] - self.learning_rate * \
                                  dW[layer]
            self.bias[layer] = self.bias[layer] - self.learning_rate * \
                               db[layer]

    def predict(self, X, y) -> Tuple[List, List]:
        """
        Predict with trained model
        :param X: Datas
        :param y: Expected values
        :return: Tuple from 2 lists with accuracies and predictions
        """
        accuracies = []
        predictions = []
        for _X, _y in zip(X.values, y.values):
            # Forward propagation
            _, A = self.prop_forward(_X, _y)
            accuracies.append(calculate_accuracy(A, _y))
            predictions.append(A)
        return accuracies, predictions

    def train(self, X_train, y_train, X_test, y_test):
        """
        Train neuronal network to get
        :param X_train Data training set
        :param y_train Expected values training set
        :param X_test Data testing set
        :param y_test Expected values testing set
        :return: List of training errors and list of test errors
        """
        training_error = []
        test_error = []

        for i in range(self.epoch):
            print("Epoch %d" % i)
            _training_error = []
            _test_error = []

            # Processing training
            for X, y in zip(X_train.values, y_train.values):
                err, _ = self.prop_forward(X, y)
                _training_error.append(err)
                self.prop_backward(X, y)

            # Processing testing
            for X, y in zip(X_test.values, y_test.values):
                err, _ = self.prop_forward(X, y)
                _test_error.append(err)

            # Add errors to lists
            training_error.append(np.mean(_training_error))
            test_error.append(np.mean(_test_error))

            # Shuffle datasets
            X_train, y_train = shuffle(X_train, y_train)
            X_test, y_test = shuffle(X_test, y_test)

        return training_error, test_error
