import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from Model.NeuralNetworkAutoDifferentiation import Neural_Network


def encode_labels(labels):
    # encode 'Adelie' as 0, 'Chinstrap' as 1, 'Gentoo' as 2
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # encode 0 as [1, 0, 0], 1 as [0, 1, 0], and 2 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded

def initialise_weights_bias(layers_nodes_num_list: list[int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    This method generates a list of layers' weights and a list of layers' bias.

    Parameter:
    layers_nodes_num_list list[int]: A list of integers that indicates how many weights and bias nodes per layer

    Outputs:
    weights list[torch.Tensor]: A list of layers' weights
    bias list[torch.Tensor]: A list of layers' bias
    """

    weights = []
    bias = []
    prev_dim = layers_nodes_num_list[0]

    for ith_layer in range(1, len(layers_nodes_num_list)):
        curr_dim = layers_nodes_num_list[ith_layer]
        initialise_weights = torch.randn(prev_dim, curr_dim, requires_grad=True)
        initialise_bias = torch.randn(1, curr_dim, requires_grad=True)
        weights.append(initialise_weights)
        bias.append(initialise_bias)

        prev_dim = curr_dim

    return weights, bias

def two_layers_model(
        activation_function: str,
        X_train: np.array, onehot_y_train: np.array,
        X_validation: np.array, integer_y_validation: np.array, 
        X_test: np.array, integer_y_test: np.array
) -> tuple[list[np.float64], list[np.float64], np.float64]:
    """
    This method train the Neural Network model with fixed layers and options of activation function based on the parameters.

    Parameters:
    activation_function str: What activation function to use. Options: sigmoid, relu, tanh
    X_train np.array: Training set
    onehot_y_train np.array: True labels of the training set
    X_validation np.array: Validation set
    integer_y_validation np.array: True labels of the validation set
    X_test np.array: Testing set
    integer_y_test np.array: True labels of the testing set

    Outputs:
    train_history list[np.float64]: A list of training accuracy through out the training
    validation_history list[np.float64]: A list of validation accuracy through out the training
    test_acc np.float64: Testing accuracy
    """
    # Parameters. As per the handout.
    learning_rate = 0.2

    w_0 = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    w_1 = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])
    b_0 = np.array([-0.02, -0.20])
    b_1 = np.array([-0.33, 0.26, 0.06])
    b_0 = b_0.reshape((1, b_0.shape[0]))
    b_1 = b_1.reshape((1, b_1.shape[0]))
    w_0 = torch.tensor(w_0, dtype=torch.float32, requires_grad=True)
    w_1 = torch.tensor(w_1, dtype=torch.float32, requires_grad=True)
    b_0 = torch.tensor(b_0, dtype=torch.float32, requires_grad=True)
    b_1 = torch.tensor(b_1, dtype=torch.float32, requires_grad=True)

    nn: Neural_Network = None
    if activation_function.__eq__("sigmoid"):
        nn = Neural_Network(learning_rate, [w_0, w_1], [b_0, b_1], "sigmoid")
    elif activation_function.__eq__("relu"):
        nn = Neural_Network(learning_rate, [w_0, w_1], [b_0, b_1], "relu")
    elif activation_function.__eq__("tanh"):
        nn = Neural_Network(learning_rate, [w_0, w_1], [b_0, b_1], "tanh")
    
    print(f"Train Neural Network with 2 layers and {activation_function} activation function")
    # Train the model 100 epochs
    train_history, validation_history = nn.train(
        X_train, onehot_y_train, 100, X_validation, integer_y_validation
    )

    # Compute and print the test accuracy
    test_predictions = nn.predict(X_test)
    test_acc = nn.accuracy(test_predictions, integer_y_test)

    return train_history, validation_history, test_acc

def multi_layers_model(
    activation_function: str, layer_nodes_list: list[int], 
    X_train: np.array, onehot_y_train: np.array,
    X_validation: np.array, integer_y_validation: np.array, 
    X_test: np.array, integer_y_test: np.array
) -> tuple[list[np.float64], list[np.float64], np.float64]:
    """
    This method train the Neural Network model with option of layers and activation functions based on the parameters.

    Parameters:
    activation_function str: What activation function to use. Options: sigmoid, relu, tanh
    layer_nodes_list list[int]: A list of integers indicating how many nodes per layer
    X_train np.array: Training set
    onehot_y_train np.array: True labels of the training set
    X_validation np.array: Validation set
    integer_y_validation np.array: True labels of the validation set
    X_test np.array: Testing set
    integer_y_test np.array: True labels of the testing set

    Outputs:
    train_history list[np.float64]: A list of training accuracy through out the training
    validation_history list[np.float64]: A list of validation accuracy through out the training
    test_acc np.float64: Testing accuracy
    """
    
    learning_rate = 0.2

    initial_weights, initial_bias = initialise_weights_bias(layer_nodes_list)

    nn: Neural_Network = None
    if activation_function.__eq__("sigmoid"):
        nn = Neural_Network(learning_rate, initial_weights, initial_bias, "sigmoid")
    elif activation_function.__eq__("relu"):
        nn = Neural_Network(learning_rate, initial_weights, initial_bias, "relu")
    elif activation_function.__eq__("tanh"):
        nn = Neural_Network(learning_rate, initial_weights, initial_bias, "tanh")

    print(f"Training Neural Network model with {len(initial_weights)} layers and {activation_function} activation function")
    # Train the model 100 epochs
    train_history, validation_history = nn.train(
        X_train, onehot_y_train, 100, X_validation, integer_y_validation
    )

    # Compute and print the test accuracy
    test_predictions = nn.predict(X_test)
    test_acc = nn.accuracy(test_predictions, integer_y_test)

    return train_history, validation_history, test_acc

def plot_results(
        results: list[list[np.float64]],
        plot_legend: list[str],
        save_file_name: str
) -> None:
    """
    This method plots the training histories of different model performances.

    Parameters:
    results list[list[np.float64]]: A slit of models' training histories
    plot_legend list[str]: A list of string indicating different models
    save_file_name str: The file name for saving the plotting figures
    """
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    plt.figure(figsize=[8, 6])
    for i, result in enumerate(results):
        plt.plot(np.array(result), colors[i], linewidth=3.0)

    plt.legend(plot_legend,fontsize=10, loc="upper left")
    plt.xlabel('Epochs ',fontsize=10)
    plt.ylabel('Accuracy',fontsize=10)
    plt.title('Accuracy Curves',fontsize=10)
    
    plt.savefig(save_file_name)

if __name__ == '__main__':

    current_location = Path.cwd()
    train_path = current_location.joinpath('data','penguins-train.csv')
    test_path = current_location.joinpath('data','penguins-test.csv')
    data = pd.read_csv(train_path)
    # the class label is last!
    y_train = data.iloc[:, -1]
    # seperate the data from the labels
    X_train = data.iloc[:, :-1]
    #scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_y_train, onehot_encoder, onehot_y_train = encode_labels(y_train)
    # labels = onehot_encoded

    pd_data_vd = pd.read_csv(test_path)
    y_validation = pd_data_vd.iloc[:, -1]
    X_validation = pd_data_vd.iloc[:, :-1]
    #scale the validation according to our training data.
    X_validation = scaler.transform(X_validation)
    validation_label_encoder, integer_y_validation, validation_onehot_encoder, onehot_y_validation = encode_labels(y_validation)
    
    pd_data_ts = pd.read_csv(test_path)
    y_test = pd_data_ts.iloc[:, -1]
    X_test = pd_data_ts.iloc[:, :-1]
    #scale the test according to our training data.
    X_test = scaler.transform(X_test)
    test_label_encoder, integer_y_test, test_onehot_encoder, onehot_y_test = encode_labels(y_test)

    print("Train Neural Network model with options of three different layer parameters")
    # Train the Neural Network model 100 epochs with three different layer parameters
    two_layer_sigmoid_train_hist, _, two_layer_sigmoid_test_acc = two_layers_model(
        "sigmoid", X_train, onehot_y_train,
        X_validation, integer_y_validation, X_test, integer_y_test
    )

    five_layer_sigmoid_train_hist, _,  five_layer_sigmoid_test_acc = multi_layers_model(
        "sigmoid", [4, 16, 8, 4, 2, 3], X_train, onehot_y_train,
        X_validation, integer_y_validation, X_test, integer_y_test
    )

    three_layer_sigmoid_train_hist, _, three_layer_sigmoid_test_acc = multi_layers_model(
        "sigmoid", [4, 8, 2, 3], X_train, onehot_y_train,
        X_validation, integer_y_validation, X_test, integer_y_test
    )

    # Save the training accuracies in local directory
    plot_results(
        [
            two_layer_sigmoid_train_hist, 
            five_layer_sigmoid_train_hist, 
            three_layer_sigmoid_train_hist
         ],
        [
            "Two layer model with sigmoid", 
            "Five layer model with sigmoid", 
            "Three layer model with sigmoid"
         ],
         "plot_diff_layers_comparison.png"
    )

    layers_report: str = \
    f"""
    Two layer model with Sigmoid activation function:
        - hidden layer nodes: [2, 3]
        - Test Accuracy: {two_layer_sigmoid_test_acc}
    Five layer model with Sigmoid activation function:
        - hidden layer nodes: [16, 8, 4, 2, 3]
        - Test Accuracy: {five_layer_sigmoid_test_acc}
    Three layer model with Sigmoid activation function:
        - hidden layer nodes: [8, 2, 3]
        - Test Accuracy: {three_layer_sigmoid_test_acc}
    """
    
    with open("layer_comparison.txt", "w") as lc_outfile:
        lc_outfile.write(layers_report)

    print(layers_report)

    print("------------------------------------------------------------------\n\
                                                                             \n\
                                                                             \n\
                                                                             \n\
          --------------------------------------------------------------------\
          ")
    
    print("Train Neural Network model with different activation functions")

    # Train the Neural Network model 100 epochs with three different activation functions
    sigmoid_train_hist, _, sigmoid_test_acc = two_layers_model(
        "sigmoid", X_train, onehot_y_train,
        X_validation, integer_y_validation, X_test, integer_y_test
    )

    relu_train_hist, _, relu_test_acc = two_layers_model(
        "relu", X_train, onehot_y_train,
        X_validation, integer_y_validation, X_test, integer_y_test
    )

    tanh_train_hist, _, tanh_test_acc = two_layers_model(
        "tanh", X_train, onehot_y_train,
        X_validation, integer_y_validation, X_test, integer_y_test
    )

    plot_results(
        [
            sigmoid_train_hist, 
            relu_train_hist, 
            tanh_train_hist
        ],
        [
            "Sigmoid Function", 
            "Relu Function", 
            "Tanh Function"
         ],
         "plot_diff_act_fun_comparison.png"
    )

    act_func_report: str = \
    f"""
    Two layer model with Sigmoid activation function:
        - Test Accuracy: {sigmoid_test_acc}
    Two layer model with Relu activation function:
        - Test Accuracy: {relu_test_acc}
    Two layer model with Tanh activation function:
        - Test Accuracy: {tanh_test_acc}
    """

    with open("act_func_comparison.txt", "w") as af_outfile:
        af_outfile.write(act_func_report)

    print(act_func_report)