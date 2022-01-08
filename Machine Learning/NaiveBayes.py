#!/usr/bin/python
from ParseText import ParseText
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class NaiveBayes:
    def __init__(self, train_path, test_path):
        self.training = []
        self.test = []
        self.targets = []

        print("Reading training data...")
        self.training = self.initialize(train_path)
        print("Reading test data...")
        self.test = self.initialize(test_path)

        self.prior = self.computePrior()
        self.likelihood = self.computeLikelihood()

        prediction = []
        accuracy = 0
        num_examples = float(len(self.test))
        for example in self.test:
            predicted = self.predict(example)
            prediction.append(predicted)
            if example.target == predicted:
                accuracy += 1

        accuracy /= num_examples
        print("Accuracy = ", accuracy)

    def plotDict(self, dict):
        plt.plot(dict.keys(), dict.values())
        plt.show()

    def list_nonhidden(self, dir):
        file_list = []
        for file in os.listdir(dir):
            if not file.startswith('.'):
                file_list.append(file)
        return file_list

    def initialize(self, dir):
        examples = []
        for folder in self.list_nonhidden(dir):
            path = os.path.join(dir, folder)
            text = []
            for file in self.list_nonhidden(path):
                filepath = os.path.join(path, file)
                instance = ParseText(filepath, stop_words=False)
                instance.target = folder  # assign folder name as target
                instance.filename = file
                examples.append(instance)

        return examples

    def addToDict(self, dict, key, values_list, duplicates):
        if key not in dict:
            dict[key] = list()

        if (duplicates == False):
            # Add only unique values, which are not already in dict
            dict[key].extend(x for x in set(values_list) if x not in dict[key])
        else:
            dict[key].extend(values_list)
        return dict

    def computeLikelihood(self):
        """
        Equations behind Likelihood:
           prob[class] = prob of a class occurring
           prob[word|class] = conditional prob of a word given a class
           prob[document|class] = product of all (prob(class) * prob[word|class]) in document
                                   or sum of log(prob(class) * prob[word|class])
        #   prediction = max ( prob[document|class] ) over all classes
        """
        self.training_words = {}
        self.dict = {}  # store training instances by target

        for instance in self.training:  # populate each target dict w/ training instances
            self.dict = self.addToDict(self.dict, instance.target, instance.text,
                                       duplicates=False)
            self.training_words = self.addToDict(self.training_words,
                                       instance.target, instance.text,
                                       duplicates=True)

        self.frequency = {}
        probability = {}  # store dict of word frequencies
                          # for each class (aka target)

        # Initalize prob of all words to zero
        for target in self.dict.keys():
            probs = {}  # store word frequency for each word
            freqs = {}
            for word in self.dict[target]:
                probs[word] = 0
                freqs[word] = 0
            probability[target] = probs.copy()
            self.frequency[target] = freqs.copy()

        # Calculate conditional prob of words given target
        for target in self.dict.keys():
            # Laplace smoothing w/ k = 1
            k = 1
            laplace_denom = self.laplace_denom(k, len(self.training_words[target]), len(self.targets))
            for word in self.training_words[target]: # list of all instances of training words
                self.frequency[target][word] += 1

            for word in self.dict[target]: # list of unique training words
                probability[target][word] = (float(self.frequency[target][word] + k) \
                                                                          / laplace_denom)

        return probability

    def laplace_denom(self, k, total_words, cardinality):
        return total_words + (k * cardinality)

    def predict(self, example):
        probability = {}
        for target in self.targets:
            constant = 1E-80 # add small constant to prevent undefined, log(0) errors
            probability[target] = np.log10(self.prior[target] + constant) # This was 0 before
            k = 1
            laplace = k / self.laplace_denom(k, len(self.dict[target]), len(self.targets))

            for word in example.text:
                if word not in self.likelihood[target].keys():  # add word with 0 frequency,
                    self.likelihood[target][word] = laplace     # and laplace-smoothed
                probability[target] += np.log10(self.likelihood[target][word] + constant)

        max_prob = float("-inf")
        predicted_class = None
        for target in self.targets: # Find target w/ the highest probability
            if probability[target] > max_prob:
                max_prob = probability[target]
                predicted_class = target

        return predicted_class

    def computePrior(self):
        # Compute prior, the probability of each target (class) occurring
        targets = []
        for instance in self.training:
            targets.append(instance.target)
        self.targets = set(targets)

        total = float(len(self.training))
        prior = {}
        for target in self.targets:
            prior[target] = 0
            count = 0
            for i in range(len(targets)):
                if target == targets[i]:
                    count += 1

            prior[target] = count / total
        return prior

if __name__ == "__main__":
    if len(sys.argv) == 1:
        current_dir = os.getcwd()
        train_path = os.path.join(current_dir, "spam-ham/train")
        test_path = os.path.join(current_dir, "spam-ham/test")
        model = NaiveBayes(train_path, test_path)

    elif len(sys.argv) == 3:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        model = NaiveBayes(train_path, test_path)

    else:
        print("Error: Expected ./NaiveBayes.py  OR ./NaiveBayes.py <train_path> <test_path>\n")
