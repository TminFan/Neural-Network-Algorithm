import torch
import numpy as np
import matplotlib.pyplot as plt

class Neural_Network:
    # Initialize the network
    def __init__(self,
                 learning_rate: float,
                 weights_list: list[torch.Tensor],
                 bias_list: list[torch.Tensor],
                 activation_func: str
                 ):

        self.weights_list = weights_list
        self.bias_list = bias_list
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.train_acc_history: list[np.float64] = []
        self.validation_acc_history: list[np.float64] = []

    def relu(self, z: torch.Tensor) -> torch.Tensor:
        """
        This method computes outputs using the RELU function

        Parameter:
        z torch.Tensor: A torch tensor output of the linear combination of the previous layer and the weights as well as bias

        Output:
        z torch.Tensor: A same shape tensor as the input tensor
        """

        rows = z.shape[0]
        cols = z.shape[1]
        intermediate_tensor = torch.zeros(cols, cols)

        for row_i in range(rows):
            for col_i in range(cols):
                original_val = z[row_i, col_i]
                # output max value between 0 and the value in the input tensor
                result = max(0, original_val)
                # diagonal value set 0 if result is not the value in the input tensor. Otherwise is 1.
                if result != original_val:
                    intermediate_tensor[row_i, row_i]=0
                else:
                    intermediate_tensor[row_i, row_i]=1
        z = torch.matmul(z, intermediate_tensor)
        
        return z
    
    def tanh(self, z: torch.Tensor) -> torch.Tensor:
        """
        This method computes outputs using Tanh function

        Parameter:
        z torch.Tensor: A torch tensor output of the linear combination of the previous layer and the weights as well as bias

        Output:
        z torch.Tensor: A same shape tensor as the input tensor
        """

        pow = torch.mul(2, z)
        sihn = torch.sub(
            torch.exp(pow),
            1
        )
        cohn = torch.add(
            torch.exp(pow),
            1
        )
        tanh = torch.divide(
            sihn,
            cohn
        )

        return tanh

    # Calculate neuron activation for an input
    def sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        """
        parameter:
        z torch.Tensor: A torch tensor output of the linear combination of the previous layer and the weights as well as bias.
        The tensor shape is (1, n_weight_nodes).

        Output:
        z a torch.Tensor: sigmoid function result value, ranging from 0 to 1.
        """

        a = torch.divide(
            1,
            torch.add(
                1, 
                torch.exp(
                    torch.sub(
                        1,
                        z
                    )
                )
            )
        )

        return a

    # Feed forward pass input to a network output
    def forward_feeding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        x torch.Tensor: n_features 2-d matrix. Shape is (4,1)

        Outputs:
        a_1 torch.Tensor: n_hidden_nodes 1-d matrix. Shape is (2,) 
        output_layer_outputs np.array: n_output_nodes 1-d matrix. Shape is (3,)
        """
        a: torch.tensor = None
        # first hidden layer
        z_0 = torch.matmul(
            torch.transpose(x, 0, 1),
            self.weights_list[0]
        ) + self.bias_list[0]
        # a should be shape (1, 2)
        if self.activation_func.__eq__("sigmoid"):
            a = self.sigmoid(z_0)
        elif self.activation_func.__eq__("relu"):
            a = self.relu(z_0)
        elif self.activation_func.__eq__("tanh"):
            a = self.tanh(z_0)

        for i in range(1, len(self.weights_list)):
            z = torch.matmul(
                a,
                self.weights_list[i]
            ) + self.bias_list[i]
            # a should be shape (1, n_output_nodes)
            if self.activation_func.__eq__("sigmoid"):
                a = self.sigmoid(z)
            elif self.activation_func.__eq__("relu"):
                a = self.relu(z)
            elif self.activation_func.__eq__("tanh"):
                a = self.tanh(z)

        return a

    # Backpropagate error and store in neurons
    def backward_propagation(
           self, x: np.array, true_y: np.array
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        This method computes the gradient values of weights and bias.

        Parameters:
        x np.array: one data point instance. n_features matrix. Shape is (4,).
        true_y np.array: hidden layer sigmoid results. n_hidden_nodes 1-d matrix. Shape is (2,).

        Outputs:
        derivative_list list[torch.Tensor]: a list containing delta values of weights and bias.
        a torch.Tensor: model prediction output given a data point
        """

        x = x.reshape((x.shape[0], 1))
        x = torch.tensor(x, dtype=torch.float32)
        true_y = true_y.reshape((1, true_y.shape[0]))
        true_y = torch.tensor(true_y, dtype=torch.float32)
        # computes classification outputs through forward feeding
        a = self.forward_feeding(x)
        # computes loss
        loss_fun = torch.sum(
            torch.multiply(
                (1/2),
                torch.square(
                    torch.sub(true_y, a)
                )
            )
        )

        # call auto differentiation
        loss_fun.backward()
        
        derivative_weight_list = []
        derivative_bias_list = []

        # record delta values
        for i in range(len(self.weights_list)):
            derivative_weight_list.append(self.weights_list[i].grad)
            derivative_bias_list.append(self.bias_list[i].grad)
        
        derivative_list = derivative_weight_list + derivative_bias_list

        return derivative_list, a
    
    def update_weights_bias(self, derivative_list: list[torch.Tensor]) -> None:
        """
        This method updates the weights and the bias values

        Parameters:
        derivative_list list[torch.Tensor]: A list of delta values of weights and bias in all layers
        """

        delta_weights_list = derivative_list[:int(len(derivative_list) / 2)]
        delta_bias_list = derivative_list[int(len(derivative_list) / 2):]

        with torch.no_grad():
            for i in range(len(delta_weights_list)):
                self.weights_list[i] -= self.learning_rate * delta_weights_list[i]
                self.bias_list[i] -= self.learning_rate * delta_bias_list[i]
                # clear the accumulated gradient values
                self.weights_list[i].grad.zero_()
                self.bias_list[i].grad.zero_()

    def train(self,
              train_instances: np.array, desired_outputs: np.array, 
              epochs: int, 
              validation_instances: np.array, validation_outputs: np.array
              ) -> tuple[list[np.float64], list[np.float64]]:
        """
        This method trains the Neural Network model for the given number of epochs and calculates training and validation accuracy in each epoch.

        Parameters:
        instances np.array: The training set. m_examples x n_features matrix. 
                Shape is (268, 4). One instance shape is (4,).
        desire_outputs np.array: The true labels of the training set. 
                m_examples x n_outputs matrix. Shape is (268, 3). One desire_output shape is (3,).
        epochs int: The number of training steps
        validation_instances np.array: The validation set. m_examples x n_features matrix. 
        validation_outputs np.array: The true labels of the validation set. 
                m_examples x n_outputs matrix.

        Outputs:
        train_acc_history list[np.float64]: Training accuracy.
        validation_acc_history list[np.float64]: Validation accuracy
        """

        for epoch in range(epochs):
            train_predictions = []
            for i, instance in enumerate(train_instances):
                true_y = desired_outputs[i]
                derivative_list, output_layer_outputs = self.backward_propagation(instance, true_y)
                output_layer_outputs = output_layer_outputs.detach().numpy()
                output_layer_outputs = output_layer_outputs.flatten()
                # convert probabilities to whole number 0, 1, 2 representing the classes
                predicted_class = np.argmax(output_layer_outputs) # index of the max value
                train_predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights_bias(derivative_list)
            
            # Calculate training and validation accuracy
            train_predictions_arr = np.array(train_predictions, ndmin=2).T # Shape is (m_examples, 1)
            desired_outputs_class = np.argmax(desired_outputs, axis=1)
            desired_outputs_class = np.reshape(desired_outputs_class, (desired_outputs_class.shape[0], 1))
            substract_diff = np.where(
                train_predictions_arr==desired_outputs_class, 1, 0
            )
            diff_sum = np.sum(substract_diff, axis=0)
            train_acc = round((diff_sum[0] / substract_diff.shape[0]), 2)
            self.train_acc_history.append(train_acc)
            # only print the last epoch train accuracy
            if epoch==99:
                epoch_100 = epoch + 1
                print(f'{epoch_100} train acc = {train_acc}')

            validation_prediction = self.predict(validation_instances)
            validation_acc = self.accuracy(validation_prediction, validation_outputs)
            self.validation_acc_history.append(validation_acc)
            
        return  self.train_acc_history, self.validation_acc_history

    def predict(self, instances: np.array) -> np.array:
        """
        This method predicts classes with given unseen data.

        Parameter:
        instances np.array: Unseen data. Shape is m_examples x n_features matrix.

        Output:
        predictions np.array: Predicted labels. Shape is m_examples 1-d matrix
        """
        predictions = []
        for instance in instances:
            instance = instance.reshape((instance.shape[0], 1))
            instance = torch.tensor(instance, dtype=torch.float32) # x shape (4, 1)
            output_layer_outputs = self.forward_feeding(instance)
            output_layer_outputs = output_layer_outputs.detach().numpy()
            output_layer_outputs = output_layer_outputs.flatten()
            predicted_class = np.argmax(output_layer_outputs)  # labels should be 0, 1, or 2.
            predictions.append(predicted_class)
        predictions = np.array(predictions)

        return predictions
    
    def accuracy(self, predictions: list[np.array], true_labels: list[np.array]) -> np.float64:
        """
        This method computes accuracy given a list of predictions and a list of true labels

        Parameters:
        predictions list[np.array]: m_examples matrix 1-d matrix
        true_labels list[np.array]: m_examples x 1 2-d matrix

        output:
        acc np.float64: Accuracy value
        """
        substract_diff = np.where(
                predictions==true_labels.flatten(), 1, 0
        )
        diff_sum = np.sum(substract_diff, axis=0)
        acc: np.float64 = round((diff_sum / substract_diff.shape[0]), 2)

        return acc