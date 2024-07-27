import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from Model.NeuralNetwork import Neural_Network


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

def run(
        n_in: int, n_hidden: int, n_out: int,
        initial_hidden_layer_weights: np.array, initial_output_layer_weights: np.array,
        initial_hidden_layer_bias: np.array, initial_output_layer_bias: np.array,
        learning_rate: float, testing_set: pd.DataFrame
) -> None:
    """
    This method train the Neural Network model and outputs the testing accuracy

    Parameters:
    n_in int: The number of input data features
    n_hidden int: The number of nodes in hidden layer
    n_out int: The number of nodes in output layer. The number of classification classes
    initial_hidden_layer_weights np.array: Hidden layer weights
    initial_output_layer_weights np.array: Output layer weights
    initial_hidden_layer_bias: Hidden layer bias
    initial_output_layer_bias: Output layer bias
    learning_rate float: learning rate for putting weights on how much updates should be put on weights and bias
    testing_set pd.DataFrame: testing data set
    """
    
    nn = Neural_Network(n_in, n_hidden, n_out, 
                        initial_hidden_layer_weights, initial_output_layer_weights, 
                        initial_hidden_layer_bias, initial_output_layer_bias,
                        learning_rate)

    print('First instance has label {}, which is {} as an integer, and {} as a list of outputs.\n'.format(
        labels[0], integer_encoded[0], onehot_encoded[0]))

    # need to wrap it into a 2D array
    instance1_prediction = nn.predict([instances[0]])
    if instance1_prediction[0] is None:
        # This should never happen once you have implemented the feedforward.
        instance1_predicted_label = "???"
    else:
        instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction)
    print('Predicted label for the first instance is: {}\n'.format(instance1_predicted_label))

    # Perform a single backpropagation pass using the first instance only. (In other words, train with 1
    #  instance for 1 epoch!). Hint: you will need to first get the weights from a forward pass.
    hidden_layer_outputs, output_layer_outputs = nn.forward_pass(instances[0])
    delta_output_layer_weights, \
    delta_hidden_layer_weights,\
    delta_output_layer_bias, \
    delta_hidden_layer_bias = nn.backward_propagate_error(
        instances[0],
        hidden_layer_outputs,
        output_layer_outputs,
        onehot_encoded[0]
    )
    nn.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)
    if nn.validate_zero_array(initial_hidden_layer_bias) and nn.validate_zero_array(initial_output_layer_bias):
        nn.update_bias(delta_output_layer_bias, delta_hidden_layer_bias)
    print('Weights after performing BP for first instance only:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)
    print('Hidden layer bias:\n', nn.hidden_layer_bias)
    print('Output layer bias:\n', nn.output_layer_bias)
    
    # Validation data
    pd_data_vd = testing_set
    validation_labels = pd_data_vd.iloc[:, -1]
    validation_instances = pd_data_vd.iloc[:, :-1]
    #scale the validation data.
    validation_instances = scaler.transform(validation_instances)
    _, validation_integer_encoded, _, _ = encode_labels(validation_labels)

    # Train for 100 epochs, on all instances.
    print(f"Train the Neural Network for 100 epochs")
    nn.train(instances, onehot_encoded, 100, validation_instances, validation_integer_encoded)
    print('\nAfter training:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    # Test data
    pd_data_ts = testing_set
    test_labels = pd_data_ts.iloc[:, -1]
    test_instances = pd_data_ts.iloc[:, :-1]
    #scale the test according to our training data.
    test_instances = scaler.transform(test_instances)
    _, test_integer_encoded, _, _ = encode_labels(test_labels)

    # Compute and print the test accuracy
    test_predictions = nn.predict(test_instances)
    test_acc = nn.accuracy(test_predictions, test_integer_encoded)
    print(f"test acc: {test_acc}")


if __name__ == '__main__':

    current_location = Path.cwd()
    train_path = current_location.joinpath('data','penguins-train.csv')
    test_path = current_location.joinpath('data','penguins-test.csv')
    data = pd.read_csv(train_path)
    testing_data = pd.read_csv(test_path)
    # the class label is last!
    labels = data.iloc[:, -1]
    # seperate the data from the labels
    instances = data.iloc[:, :-1]
    #scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    instances = scaler.fit_transform(instances)
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    # labels = onehot_encoded

    # Parameters. As per the handout.
    n_in = 4
    n_hidden = 2
    n_out = 3
    learning_rate = 0.2

    initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])
    zero_hidden_layer_bias = np.zeros((n_hidden))
    zero_output_layer_bias = np.zeros((n_out))
    
    print("Run Neural Network without bias")
    run(
        n_in, n_hidden, n_out, 
        initial_hidden_layer_weights, initial_output_layer_weights,
        zero_hidden_layer_bias, zero_output_layer_bias, learning_rate, testing_data
    )

    print("------------------------------------------------------------------\n\
                                                                             \n\
                                                                             \n\
                                                                             \n\
          --------------------------------------------------------------------\
          ")

    print("Run Neural Network with bias")
    initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])
    initial_hidden_layer_bias = np.array([-0.02, -0.20])
    initial_output_layer_bias = np.array([-0.33, 0.26, 0.06])

    run(
        n_in, n_hidden, n_out, 
        initial_hidden_layer_weights, initial_output_layer_weights,
        initial_hidden_layer_bias, initial_output_layer_bias, learning_rate
    )