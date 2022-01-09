#!/usr/bin/python
from ParseText import ParseText
import numpy as np
import random
import os
import sys


class Perceptron:
    def __init__(self, train_filepath, test_filepath, activation):
        self.targets = []
        self.output = None

        self.weights = {}
        self.feature_vals = {}
        self.total_weights = {}
        self.total_features = {}
        self.activation_fn = activation

        print("Reading training data...")
        train_data = self.initialize(train_filepath)
        print("Reading test data...")
        test_data = self.initialize(train_filepath)
        print("Training...")
        self.train(train_data)
        prediction = self.predict(test_data)
        accuracy = self.compute_accuracy(test_data, prediction)
        print("Accuracy: ", accuracy)

    @staticmethod
    def list_nonhidden(directory):
        file_list = []
        for file in os.listdir(directory):
            if not file.startswith('.'):
                file_list.append(file)
        return file_list

    @staticmethod
    def normalize(array, new_min, new_max):
        old_min = np.min(array)
        old_max = np.max(array)
        old_range = old_max - old_min
        for i in range(len(array)):
            array[i] = new_min + ((array[i] - old_min) * (new_max - new_min)) / old_range
        return array

    def initialize(self, directory):
        examples = []
        for folder in self.list_nonhidden(directory):
            path = directory + "//" + folder
            for file in self.list_nonhidden(path):
                instance = ParseText(path + "//" + file, stop_words=False)
                instance.target = folder  # assign target (label/string) to instance
                if instance.target not in self.targets:
                    self.targets.append(instance.target)    # add all possible values
                                                            # for target into target vector
                instance.filename = file
                examples.append(instance)

        # Assign numerical values to each class (binomial)
        for instance in examples:
            if instance.target == self.targets[0]:
                instance.target = 1
            else:
                if self.activation_fn == "linear":
                    instance.target = -1
                if self.activation_fn == "sigmoid":
                    instance.target = 0

        return examples

    @staticmethod
    def sum_unit(weights, feature_vals, example):
        """
        Compute output from (linear) sum unit
        :returns sum: (1xn array) list of floats for each training example.
                    where each element is the sum of (weights[i] * feature_vals[i])
                    for i = 0 to len(weights)-1
        """
        summation = 0
        for key in example.vocabulary:
            if key in weights.keys() and key in feature_vals.keys():
                summation += weights[key] * feature_vals[key]

        summation += weights["_bias"] * feature_vals["_bias"]
        return summation

    @staticmethod
    def sign_unit(input):
        """
        Calculate output from sign activation function (return +/- class)
        :param input: output from sum_unit for each training example

        :returns: +1 if input >= 0, -1 if input < 0
        """
        output = np.zeros_like(input)
        for i in range(len(input)):  # For each training example
            output[i] = -1  # negative class
            if input[i] >= 0:
                output[i] = 1  # positive class
        return output

    def sigmoid_unit(self, input):
        """
        Calculate output from sigmoid activation function.

        :param: input: (float) output from sum_unit for each training example
        :returns: (float) sigmoid(input) = 1 / [1 + exp(-sum(input))]

        Note:
            - One downside of sigmoid fn is the tendency for the gradient to approach
            zero or infinity (vanishing and exploding gradients, respectively).
            When gradient goes to inf, weight updates have too much variance and
            when gradient goes to zero, weight updates stop entirely.

            - One way to avoid this is with batch normalization, which limits the
            output of the sigmoid_unit to a narrower range of the sigmoid function
            (excluding the outer tails near 0 and 1).
        """
        output = np.zeros_like(input)
        for i in range(len(input)):
            net = np.exp(-1 * input[i])
            output[i] = 1.0 / (1.0 + net)

        output = self.normalize(output, 0.3, 0.8)
        return output

    @staticmethod
    def derive_sigmoid(sigmoid):
        """
        Calculate derivative of sigmoid function.

        :param: sigmoid: (float) output from the sigmoid function
        :returns: derivative of sigmoid
        """
        derivative = float(sigmoid) * (1.0 - float(sigmoid))
        return derivative

    def gradient_descent(self, learning_rate, descent, train_data, output):
        summation = np.zeros_like(train_data)

        self.update_weights(learning_rate, train_data, output)
        for i, example in enumerate(train_data):
            summation[i] = self.sum_unit(self.weights, self.feature_vals, example)

            # stochastic gradient descent
            if descent == "stochastic":
                output = self.activation(summation, self.activation_fn)

        # Batch gradient descent
        if descent == "batch":
            output = self.activation(summation, self.activation_fn)

        return output

    def activation(self, input, activation):
        """
        Update predicted class (output) for each example based on activation function

        :param: input: (1xn array) output from sum unit for each training example
        :param: activation: (str) indicating type of activation function to use
        :returns: (1xn array) output from activation unit
        """
        output = np.zeros_like(input)
        if activation == "linear":
            output = self.sign_unit(input)

        if activation == "sigmoid":
            output = self.sigmoid_unit(input)

        return output

    def update_weights(self, learning_rate, train_data, output):
        """
        Update the weights for each training example based on prediction error.

        :param: learning_rate: (float) how quickly weights are updated
        :param: train_data: (1xn array) list of ParseText objects
        :param: output: (1xn array) list of integers, indicating predicted class

        :returns: None
        """
        for i, example in enumerate(train_data):
            for feature in example.vocabulary:  # For number of features in training example
                if self.activation_fn == "linear":
                    error_term = example.target - output[i]

                if self.activation_fn == "sigmoid":
                    error_term = (example.target - output[i]) \
                                    * self.derive_sigmoid(output[i])  # was output before

                else:
                    print("Must specify activation function.")
                    return

                # Add the weight update (if error term = 0, no weight update)
                if i == 0:  # Bias term
                    self.weights[feature] += learning_rate * self.feature_vals[feature]
                else:
                    self.weights[feature] += learning_rate * error_term * self.feature_vals[feature]

    @staticmethod
    def compute_loss(actual, predicted, metric):
        """
        Calculate loss from actual and predicted values.

        :param: actual: (1xn array) list of classes for each train/test example
        :param: predicted: (1xn array) list of predicted classes for each example
        :param: metric: (str) indicating which loss metric to use

        Loss functions:
          mse: sum of (y_actual - y_predic)^2 over all examples

          cross entropy: -[ sum of (y_actual * log(y_predic) +
                         (1-y_actual) * log(1-y_predic)) ] over all examples

        :returns: error computed by loss function
        """
        error = 0
        constant = 1E-80
        for i in range(len(actual)):
            if metric == "mse":  # mean-squared error
                error += 0.5 * (actual[i].target - predicted[i]) ** 2

            if metric == "cross entropy":  # binary cross entropy
                error -= (actual[i].target * np.log10(predicted[i] + constant)) + \
                                ((1-actual[i].target) * np.log10(1-predicted[i] + constant))

        return error

    def train(self, train_data):
        """
        Iteratively update weights until the output from the activation unit
            converges (weights stop updating) on an ideal prediction for all examples.

        Update the feature weights using an error metric equal to:
            weight_update = prev_weight * error_term * feature_val

        Weights for sigmoid are initialized using Xavier Initialization of zero-mean,
        normally distributed values between [-1/sqrt(n), 1/sqrt(n)]
        with a variance of 1/n, where n is the number of input neurons (in this case, 1).

        :param train_data = 1xn array of ParseText objects
        :returns: None
        """
        output = np.zeros_like(train_data)   # No +1 element; bias has no true/predicted values
        summation = np.zeros_like(train_data)

        # Initialize weight vector to random values and store feature vector values
        # NOTE: One weight vector and one feature vector per training example
        for i, example in enumerate(train_data):
            # The perceptron bias term is represented by "_bias" to differentiate it from a
            # training/test word of "bias"
            if self.activation_fn == "linear":
                self.weights["_bias"] = random.uniform(0.01, 0.5)   # bias weight

            if self.activation_fn == "sigmoid":
                self.weights["_bias"] = 0  # bias weight

            self.feature_vals["_bias"] = 1   # bias value

            for feature in example.vocabulary:
                if self.activation_fn == "linear":
                    self.weights[feature] = random.uniform(0.01, 0.5)   # bias weight
                if self.activation_fn == "sigmoid":
                    min_weight = -1.0
                    max_weight = -1 * min_weight
                    self.weights[feature] = random.uniform(min_weight, max_weight)  # bias weight

                self.feature_vals[feature] = example.vocabulary[feature]

            # Calculate initial output from sum unit
            # NOTE: One output per sum_unit per training example
            summation[i] = self.sum_unit(self.weights, self.feature_vals, example)

        if self.activation_fn == "linear":
            output = self.sign_unit(summation)
            loss_metric = "mse"

        if self.activation_fn == "sigmoid":
            output = self.sigmoid_unit(summation)
            loss_metric = "mse"

        else:
            print("Must specify activation function.")
            return

        num_iter = 0
        learning_rate = 0.5
        min_error = np.Inf
        prev_error = 0
        delta_error = np.Inf
        while delta_error > 0.01 and num_iter < 100:  # delta_error > 0.01 # error > 0 or iter < 100:
            output = self.gradient_descent(learning_rate, "batch", train_data, output)
            error = self.compute_loss(train_data, output, loss_metric)
            # delta_error = np.abs(error - min_error)
            delta_error = np.abs(error - prev_error)
            print("error = ", error, " min error = ", min_error)
            print("delta error = ", delta_error)

            if error < min_error:
                min_error = error

            learning_rate += 0.05
            num_iter += 1

    def predict(self, test_data):
        summation = np.zeros_like(test_data)
        for i, example in enumerate(test_data):
            summation[i] = self.sum_unit(self.weights, self.feature_vals, example)

        if self.activation_fn == "linear":
            output = self.sign_unit(summation)

        if self.activation_fn == "sigmoid":
            output = self.sigmoid_unit(summation)
            output = self.threshold(output, 0.5, 0, 1)  # round values to 0 or 1
        else:
            print("Must specify activation function.")
            return

        return output

    @staticmethod
    def threshold(array, threshold, val1, val2):
        for i in range(len(array)):
            if array[i] < threshold:
                array[i] = val1
            else:
                array[i] = val2
        return array

    @staticmethod
    def compute_accuracy(test_data, prediction):
        accuracy = 0
        for i, example in enumerate(test_data):
            if example.target == prediction[i]:
                accuracy += 1

        accuracy /= len(test_data)
        return accuracy


if __name__ == "__main__":
    if len(sys.argv) == 1:
        current_dir = os.getcwd()
        train_path = os.path.join(current_dir, "spam-ham/train")
        test_path = os.path.join(current_dir, "spam-ham/test")
        activation_fn = "sigmoid"  # "linear"
        model = Perceptron(train_path, test_path, activation_fn)

    elif len(sys.argv) == 4:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        activ_fn = sys.argv[3]
        model = Perceptron(train_path, test_path, activ_fn)

    else:
        print("Error: Expected ./Perceptron.py  OR ./Perceptron.py <train_path> <test_path>\n")
