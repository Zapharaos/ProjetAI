from typing import List, Optional

import numpy as np

from utils import Utility


class Neuron:
    weights: []
    bias: []
    value: float
    error: float
    bias_diff: float
    to: Optional["Layer"]

    def __init__(self, to: Optional["Layer"]):
        self.weights = []
        self.to = to
        self.value = 0
        self.error = 0
        self.bias_diff = 0

        self.weights = [] if to is None else \
            np.random.uniform(low=-1.0, high=1.0, size=(to.size,))
        self.bias = np.random.uniform(low=-1.0, high=1.0)

    def __str__(self):
        return str(self.value)


class Layer:
    neurons: List[Neuron]
    next_layer: Optional["Layer"]
    size: int

    def __init__(self, size: int):
        self.neurons = []
        self.size = size
        self.next_layer = None

    def initialize(self, next_layer: "Layer"):
        self.next_layer = next_layer

        for i in range(self.size):
            self.neurons.append(Neuron(to=next_layer))

    def prop_forward(self):
        if self.next_layer is None:  # Last layer

            softmax = Utility.softmax([n.value for n in self.neurons])

            for i in range(len(softmax)):
                self.neurons[i].value = softmax[i]

            return

        for i in range(len(self.next_layer.neurons)):
            neuron = self.next_layer.neurons[i]
            value = sum([n.value * n.weights[i] for n in
                         self.neurons]) + neuron.bias

            # if next_layer is last layer
            if self.next_layer.next_layer is None:
                neuron.value = value
            else:
                neuron.value = np.tanh(value)

    def prop_backward(self):
        pass

    def __str__(self):
        return ", ".join([str(a) for a in self.neurons])


class NeuralNet:
    layers: List[Layer]

    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None,
                 hidden_layer_sizes=(4,), activation='identity',
                 learning_rate=0.1, epoch=200):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.n_layers = len(hidden_layer_sizes)
        self.layers = []
        X = X_train.to_numpy().transpose()
        y = y_train.to_numpy().transpose()

        # start layer
        self.layers.append(Layer(size=len(X)))

        for i in range(self.n_layers):
            self.layers.append(Layer(size=hidden_layer_sizes[i]))

        # end layer
        self.layers.append(Layer(size=len(y)))

        layers_len = len(self.layers)
        for i in range(layers_len):
            self.layers[i].initialize(
                next_layer=self.layers[i + 1] if i + 1 < layers_len else None
            )

    def prop_forward(self, X):
        # Z (ponderations)
        # W (weights)
        # A (activations)
        # b (biais)
        # premiere couche
        for j in range(0, len(X[0])):
            # init first layers neurons
            for i in range(len(X)):
                self.layers[0].neurons[i].value = X[i][j]

            for layer in self.layers:
                layer.prop_forward()

    def prop_backward(self, X, y, momentum: int = 1):
        for layer in reversed(self.layers):
            for neuron in layer.neurons:
                neuron.bias_diff = self.learning_rate * neuron.error + momentum * neuron.bias_diff
                neuron.bias += neuron.bias_diff




    def calculate_errors(self, y, idx):
        if len(self.layers) == 0:
            return

        last_layer = self.layers[-1]

        for i in range(len(last_layer.neurons)):
            neuron = last_layer.neurons[i]

            neuron.error = (y[i][idx] - neuron.value) * neuron.value * (
                    1 - neuron.value)

        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            for j in range(len(layer.neurons)):
                n1 = layer.neurons[j]
                err = 0

                for n2 in next_layer.neurons:
                    err += n2.weights[j] * n2.weights

                n1.error = n1.value * (1 - n1.value) * err

    def train(self, X, y):
        # For each input
        for j in range(len(y)):
            X_data = X[j]
            # Prop forward

            # Set all values
            for i in range(len(X_data)):
                self.layers[0].neurons[i].value = X_data[i]

            # Calculate weights and activations
            for layer in self.layers:
                layer.prop_forward()

            self.calculate_errors(y, j)
            self.prop_backward(X, y)

    # for y_val in y:

    def __str__(self):
        return "\n".join([str(a) for a in self.layers])
