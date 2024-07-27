import numpy as np
import math
import matplotlib.pyplot as plt

class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, 
                 hidden_layer_weights, output_layer_weights, 
                 hidden_layer_bias, output_layer_bias,
                 learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.hidden_layer_bias = hidden_layer_bias
        self.output_layer_bias = output_layer_bias

        self.learning_rate = learning_rate

        self.train_acc_history = []
        self.validation_acc_history = []

    # Calculate neuron activation for an input
    def sigmoid(self, input) -> np.float64:
        """
        This method computes outputs using the sigmoid function.

        parameter:
        input np.float64: weighted sum
        
        Output:
        output np.float64: sigmoid function result value, ranging from 0 to 1.
        """
        output = 1 / (1 + math.exp((-input)))
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs) -> tuple[np.array, np.array]:
        """
        Parameters:
        inputs np.array: n_features 1-d matrix. Shape is (4,)
        
        Outputs:
        hidden_layer_outputs np.array: n_hidden_nodes 1-d matrix. Shape is (2,) 
        output_layer_outputs np.array: n_output_nodes 1-d matrix. Shape is (3,)
        """
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = np.matmul(
                inputs, self.hidden_layer_weights[:, i]
            )
            if np.count_nonzero(self.hidden_layer_bias[i])!=0:
                weighted_sum += self.hidden_layer_bias[i]
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)
        hidden_layer_outputs = np.array(hidden_layer_outputs)
        
        output_layer_outputs = []
        for i in range(self.num_outputs):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = np.matmul(
                hidden_layer_outputs, self.output_layer_weights[:, i]
            )
            if np.count_nonzero(self.output_layer_bias[i])!=0:
                weighted_sum += self.output_layer_bias[i]
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)
        output_layer_outputs = np.array(output_layer_outputs)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(
            self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs
    ) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Parameters:
        inputs np.array: one data point instance. n_features matrix. Shape is (4,).
        hidden_layer_outputs np.array: hidden layer sigmoid results. n_hidden_nodes 1-d matrix. Shape is (2,).
        output_layer_outputs np.array: output layer sigmoid results. n_output_nodes 1-d matrix. Shape is (3,).
        desired_outputs np.array: one data point onehot_encoded array. n_outputs 1-d matrix. Shape is (3,).

        Outputs:
        delta_output_layer_weights np.array: derivative output nodes weights. n_hidden_nodes x m_output_nodes matrix. Shape is (2, 3)
        delta_hidden_layer_weights np.array: derivative hidden nodes weights. n_features x m_hidden_nodes matrix. Shape is (4, 2)
        output_layer_betas np.array: derivative output nodes bias. m_output_nodes 1-d matrix. Shape is (3, )
        hidden_layer_betas np.array: derivative hidden nodes bias. m_hidden_nodes 1-d matrix. Shape is (2, )
        """

        output_layer_betas = np.zeros(self.num_outputs) # Shape is (3,)
        # Calculate output layer betas.
        output_layer_error =  output_layer_outputs -desired_outputs
        output_layer_betas += output_layer_error *\
            output_layer_outputs * (1 - output_layer_outputs) # Shape is (3,)
        # print('OL betas: ', output_layer_betas)

        hidden_layer_betas = np.zeros(self.num_hidden)
        # Calculate hidden layer betas.
        intermediate_derivative = np.matmul(
            self.output_layer_weights, # Shape is (2, 3)
            np.reshape(output_layer_betas, (output_layer_betas.shape[0], 1)) # Shape is (3, 1)
        ) # Shape is (2, 1)
        hidden_layer_betas +=  intermediate_derivative.flatten() *\
        (hidden_layer_outputs * (1 - hidden_layer_outputs)) # Shape is (2,)
        # print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # Calculate output layer weight changes.
        delta_output_layer_weights += np.matmul(
            np.reshape(hidden_layer_outputs, (hidden_layer_outputs.shape[0], 1)), # Shape is (2, 1)
            np.reshape(output_layer_betas, (1, output_layer_betas.shape[0])) # Shape is (1, 3)
        )

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # Calculate hidden layer weight changes.
        delta_hidden_layer_weights += np.matmul(
            np.reshape(inputs, (inputs.shape[0], 1)), # Shape is (4, 1)
            np.reshape(hidden_layer_betas, (1, hidden_layer_betas.shape[0])) # Shape is (1, 2)
        )

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights, output_layer_betas, hidden_layer_betas

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights) -> None:
        """
        This method updates weights.

        Parameters:
        delta_output_layer_weights np.array: derivative output nodes weights. n_hidden_nodes x m_output_nodes matrix. Shape is (2, 3)
        delta_hidden_layer_weights np.array: derivative hidden nodes weights. n_features x m_hidden_nodes matrix. Shape is (4, 2)
        """
        # Update the weights.
        self.output_layer_weights -= (self.learning_rate * delta_output_layer_weights)
        self.hidden_layer_weights -= (self.learning_rate * delta_hidden_layer_weights)

    def update_bias(self, delta_output_layer_betas, delta_hidden_layer_betas) -> None:
        """
        This methods updates bias.

        Parameters:
        output_layer_betas np.array: derivative output nodes bias. m_output_nodes 1-d matrix. Shape is (3, )
        hidden_layer_betas np.array: derivative hidden nodes bias. m_hidden_nodes 1-d matrix. Shape is (2, )
        """
        self.output_layer_bias -= (self.learning_rate * delta_output_layer_betas)
        self.hidden_layer_bias -= (self.learning_rate * delta_hidden_layer_betas)

    def train(self, train_instances, desired_outputs, epochs, validation_instances, validation_outputs) -> None:
        """
        This method trains the Neural Network with given epochs

        Parameters:
        instances np.array: m_examples x n_features matrix. Shape is (268, 4). One instance shape is (4,).
        desire_outputs np.array: m_examples x n_outputs matrix. Shape is (268, 3). One desire_output shape is (3,).
        """

        for epoch in range(epochs):

            train_predictions = []
            for i, instance in enumerate(train_instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, \
                delta_hidden_layer_weights, \
                delta_output_layer_bias, \
                delta_hidden_layer_bias = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i]
                )
                # convert output probabilities into label 0, 1, 2
                predicted_class = np.argmax(output_layer_outputs) # index of the max value
                train_predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)
                if self.validate_zero_array(self.hidden_layer_bias) \
                    and self.validate_zero_array(self.output_layer_bias):
                        self.update_bias(delta_output_layer_bias, delta_hidden_layer_bias)

            # get training accuracy achieved over this epoch
            train_predictions_arr = np.array(train_predictions, ndmin=2).T # Shape is (m_examples, 1)
            desired_outputs_class = np.argmax(desired_outputs, axis=1)
            desired_outputs_class = np.reshape(desired_outputs_class, (desired_outputs_class.shape[0], 1))
            substract_diff = np.where(
                train_predictions_arr==desired_outputs_class, 1, 0
            )
            diff_sum = np.sum(substract_diff, axis=0)
            train_acc = round((diff_sum[0] / substract_diff.shape[0]), 2)
            self.train_acc_history.append(train_acc)
            
            # print accuracy every 10 epoch
            if (epoch+1) % 10 == 0:
                print(f"{epoch} train acc = {train_acc}")

            # get validation accuracy using the learnt model over this epoch
            validation_prediction = self.predict(validation_instances)
            validation_acc = self.accuracy(validation_prediction, validation_outputs)
            self.validation_acc_history.append(validation_acc)
            
        return self.train_acc_history, self.validation_acc_history

    def predict(self, instances: np.array) -> np.array:
        """
        This method predicts classes given the unseen data.

        Parameter:
        instances np.array: m_examples x n_features matrix. Shape is (268, 4). One instance shape is (4,).

        Output:
        predictions np.array: A list of predicted labels. Shape is m_examples 1-d matrix
        """
        predictions = []
        for instance in instances:
            _, output_layer_outputs = self.forward_pass(instance)
            predicted_class = np.argmax(output_layer_outputs)  # label should be 0, 1, or 2.
            predictions.append(predicted_class)
        predictions = np.array(predictions)

        return predictions
    
    def accuracy(self, predictions: np.array, true_labels: np.array) -> np.float64:
        """
        This method computes accuracy.

        Parameters:
        predictions np.array: m_examples matrix
        true_labels np.array: m_examples x 1 2-d matrix

        Output:
        acc np.float64: Accuracy
        """
        substract_diff = np.where(
                predictions==true_labels.flatten(), 1, 0
        )
        diff_sum = np.sum(substract_diff, axis=0)
        acc = round((diff_sum / substract_diff.shape[0]), 2)

        return acc
    
    def validate_zero_array(self, array_list: list[np.array]) -> bool:
        """
        This method validates if any of the arrays in the list contains zero

        Parameter:
        array_list list[np.array]: A list of bias arrays
        """
        for arr in array_list:
            if np.count_nonzero(arr):
                continue
            else:
                return False
        
        return True
    
    def plot_train_test_history(self) -> None:
        """
        This method plots training history of the model.
        """
        plt.figure(figsize=[8, 6])
        plt.plot(np.array(self.train_acc_history),'r',linewidth=3.0)
        plt.plot(np.array(self.validation_acc_history),'b',linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=10, loc="upper left")
        plt.xlabel('Epochs ',fontsize=10)
        plt.ylabel('Accuracy',fontsize=10)
        plt.title('Accuracy Curves',fontsize=10)

        plt.show()