import argparse
import os
from os import walk

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from neuronalnetwork import NeuralNet
from utils import confusion_matrix, display_confusion_matrix, \
    ClassificationReport


def process_neural_network(file_name: str,
                           error_plot_file: str = None,
                           error_plot_title: str = "Errors",
                           confusion_plot_file: str = None,
                           confusion_plot_title: str = "",
                           confusion_plot_cmap: str = "Reds",
                           activation: str = 'tanh',
                           epoch: int = 100,
                           hidden_layer_sizes=(10, 8, 6),
                           heuristic: str = 'xavier'):
    """
    Process to neuronal network training
    :param file_name: Data file
    :param error_plot_file: Error plot file
                            (if not None, plot is saved to path)
    :param error_plot_title: Error plot title
    :param confusion_plot_file: Confusion plot file
                                (if not None, plot is saved to path)
    :param confusion_plot_title: Confusion plot title
    :param confusion_plot_cmap: Confusion plot heatmap color
    :param activation: Activation function
    :param epoch: Epochs
    :param hidden_layer_sizes: Hidden layer sizes
    :param heuristic: Heuristic function for weights initialization
    """
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

    # Initialization of the model
    model = NeuralNet(X_train=X_train, y_train=y_train, X_test=X_test,
                      y_test=y_test, activation=activation, epoch=epoch,
                      hidden_layer_sizes=hidden_layer_sizes,
                      heuristic=heuristic)
    # Fitting the model with data
    train_errors, test_errors = model.train(X_train, y_train, X_test, y_test)

    accuracies, y_pred = model.predict(X_test, y_test)
    print("Accuracy: {:.2f}".format(np.mean(accuracies)))

    new_pred = []
    for pred in y_pred:
        new_pred.append(np.argmax(pred, axis=0))
    y_pred = new_pred

    y_test = np.argmax(y_test.values, axis=1)

    # Confusion matrix
    matrix = confusion_matrix(y_pred=y_pred, y_test=y_test)

    # Plotting the graph of the errors
    plt.plot(train_errors, label='train')
    plt.plot(test_errors, label='test')
    plt.legend()
    plt.title(error_plot_title)
    plt.xlabel('Epochs')
    plt.ylabel('Error')

    if error_plot_file is None:
        plt.show()
    else:
        dir_name = os.path.dirname(error_plot_file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        plt.savefig(error_plot_file)

        plt.clf()

    # Generate plot for confusion matrix
    display_confusion_matrix(matrix, ["0", "1", "2", "3"],
                             cmap=confusion_plot_cmap,
                             title=confusion_plot_title,
                             file_name=confusion_plot_file)

    # classification metrics
    report = ClassificationReport(matrix)

    report.print_metrics()


def part2():
    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/relu6-4/xavier-errors.png",
        error_plot_title="Relu 6-4 (xavier)",
        confusion_plot_file="data/images/NN/relu6-4/xavier-confusion.png",
        confusion_plot_title="Confusion matrix for relu 6-4 (xavier)",
        confusion_plot_cmap="Blues",
        activation="relu",
        hidden_layer_sizes=(6, 4)
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/relu6-4/he-et-al-errors.png",
        error_plot_title="Relu 6-4 (he-et-al)",
        confusion_plot_file="data/images/NN/relu6-4/he-et-al-confusion.png",
        confusion_plot_title="Confusion matrix for relu 6-4 (he-et-al)",
        confusion_plot_cmap="Blues",
        activation="relu",
        hidden_layer_sizes=(6, 4),
        heuristic='he-et-al'
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/tanh6-4/xavier-errors.png",
        error_plot_title="Tanh 6-4 (xavier)",
        confusion_plot_file="data/images/NN/tanh6-4/xavier-confusion.png",
        confusion_plot_title="Confusion matrix for tanh 6-4 (xavier)",
        confusion_plot_cmap="Blues",
        activation="tanh",
        hidden_layer_sizes=(6, 4)
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/tanh6-4/he-et-al-errors.png",
        error_plot_title="Tanh 6-4 (he-et-al)",
        confusion_plot_file="data/images/NN/tanh6-4/he-et-al-confusion.png",
        confusion_plot_title="Confusion matrix for tanh 6-4 (he-et-al)",
        confusion_plot_cmap="Blues",
        activation="tanh",
        hidden_layer_sizes=(6, 4),
        heuristic='he-et-al'
    )

    ###
    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/relu10-8-4/xavier-errors.png",
        error_plot_title="Relu 10-8-4 (xavier)",
        confusion_plot_file="data/images/NN/relu10-8-4/xavier-confusion.png",
        confusion_plot_title="Confusion matrix for relu 10-8-4 (xavier)",
        confusion_plot_cmap="Blues",
        activation="relu",
        hidden_layer_sizes=(10, 8, 4)
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/relu10-8-4/he-et-al-errors.png",
        error_plot_title="Relu 10-8-4 (he-et-al)",
        confusion_plot_file="data/images/NN/relu10-8-4/he-et-al-confusion.png",
        confusion_plot_title="Confusion matrix for relu 10-8-4 (he-et-al)",
        confusion_plot_cmap="Blues",
        activation="relu",
        hidden_layer_sizes=(10, 8, 4),
        heuristic='he-et-al'
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/tanh10-8-4/xavier-errors.png",
        error_plot_title="Tanh 10-8-4 (xavier)",
        confusion_plot_file="data/images/NN/tanh10-8-4/xavier-confusion.png",
        confusion_plot_title="Confusion matrix for tanh 10-8-4 (xavier)",
        confusion_plot_cmap="Blues",
        activation="tanh",
        hidden_layer_sizes=(10, 8, 4)
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/tanh10-8-4/he-et-al-errors.png",
        error_plot_title="Tanh 10-8-4 (he-et-al)",
        confusion_plot_file="data/images/NN/tanh10-8-4/he-et-al-confusion.png",
        confusion_plot_title="Confusion matrix for tanh 10-8-4 (he-et-al)",
        confusion_plot_cmap="Blues",
        activation="tanh",
        hidden_layer_sizes=(10, 8, 4),
        heuristic='he-et-al'
    )

    ###
    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/relu10-8-6/xavier-errors.png",
        error_plot_title="Relu 10-8-6 (xavier)",
        confusion_plot_file="data/images/NN/relu10-8-6/xavier-confusion.png",
        confusion_plot_title="Confusion matrix for relu 10-8-6 (xavier)",
        confusion_plot_cmap="Blues",
        activation="relu",
        hidden_layer_sizes=(10, 8, 6)
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/relu10-8-6/he-et-al-errors.png",
        error_plot_title="Relu 10-8-6 (he-et-al)",
        confusion_plot_file="data/images/NN/relu10-8-6/he-et-al-confusion.png",
        confusion_plot_title="Confusion matrix for relu 10-8-6 (he-et-al)",
        confusion_plot_cmap="Blues",
        activation="relu",
        hidden_layer_sizes=(10, 8, 6),
        heuristic='he-et-al'
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/tanh10-8-6/xavier-errors.png",
        error_plot_title="Tanh 10-8-6 (xavier)",
        confusion_plot_file="data/images/NN/tanh10-8-6/xavier-confusion.png",
        confusion_plot_title="Confusion matrix for tanh 10-8-6 (xavier)",
        confusion_plot_cmap="Blues",
        activation="tanh",
        hidden_layer_sizes=(10, 8, 6)
    )

    process_neural_network(
        file_name="data/synthetic.csv",
        error_plot_file="data/images/NN/tanh10-8-6/he-et-al-errors.png",
        error_plot_title="Tanh 10-8-6 (he-et-al)",
        confusion_plot_file="data/images/NN/tanh10-8-6/he-et-al-confusion.png",
        confusion_plot_title="Confusion matrix for tanh 10-8-6 (he-et-al)",
        confusion_plot_cmap="Blues",
        activation="tanh",
        hidden_layer_sizes=(10, 8, 6),
        heuristic='he-et-al'
    )


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

    report.print_metrics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Projet AI commands')
    parser.add_argument('mode', type=int, help="Mode: (0: Neural Network,"
                                               "1: part2, 2: part3)")
    parser.add_argument("--data",
                        default="data/synthetic.csv",
                        type=str,
                        dest="data",
                        help="Data file for neuronal network")
    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        dest="epochs",
                        help="Number of epochs")
    parser.add_argument("--activation",
                        default="tanh",
                        type=str,
                        dest="activation",
                        help="Activation function (tanh, relu, identity, "
                             "sigmoid)")
    parser.add_argument("--heuristic",
                        default="xavier",
                        type=str,
                        dest="heuristic",
                        help="Heuristic function (weights initialization): "
                             "xavier or he-et-al")
    parser.add_argument("--hidden_layer_sizes",
                        default=(6, 4),
                        type=tuple,
                        dest="hidden_layer_sizes",
                        help="Hidden layer sizes (example: (6,4))")
    parser.add_argument("--error_plot_file",
                        default=None,
                        type=str,
                        dest="error_plot_file",
                        help="Error plot file path")
    parser.add_argument("--error_plot_title",
                        default="Test and training errors / epochs",
                        type=str,
                        dest="error_plot_title",
                        help="Error plot title")
    parser.add_argument("--confusion_plot_file",
                        default=None,
                        type=str,
                        dest="confusion_plot_file",
                        help="Confusion matrix plot file")
    parser.add_argument("--confusion_plot_title",
                        default="Confusion matrix",
                        type=str,
                        dest="confusion_plot_title",
                        help="Confusion matrix plot title")
    parser.add_argument("--confusion_plot_cmap",
                        default="Reds",
                        type=str,
                        dest="confusion_plot_cmap",
                        help="Confusion matrix color map (Reds, Blues, ...)")
    args = parser.parse_args()

    if args.mode == 0:
        # Calculate Neural Network
        process_neural_network(
            file_name=args.data,
            error_plot_file=args.error_plot_file,
            error_plot_title=args.error_plot_title,
            confusion_plot_file=args.confusion_plot_file,
            confusion_plot_title=args.confusion_plot_title,
            confusion_plot_cmap=args.confusion_plot_cmap,
            activation=args.activation,
            hidden_layer_sizes=args.hidden_layer_sizes,
            heuristic=args.heuristic,
            epoch=args.epochs
        )
    elif args.mode == 1:
        part2()
    elif args.mode == 2:
        part3()
    else:
        print("Mode not recognised.")
