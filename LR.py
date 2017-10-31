import random
import numpy as np

def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))

class LogReg:
    def __init__(self, iterations=1000, learning_rate=1e5):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def predict(self, data):
        if len(data) == 0:
            raise ValueError('Expecting 2D array')

        if len(data[0]) != self.features_size:
            raise ValueError('Invalid feature size')

        predictions = []

        for example in data:
            probablity = sigmoid(np.dot(example, self.weights) - self.t)

            if probablity > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def train(self, data, labels):
        self.t = 10
        self.weights = np.zeros(len(data[0]))

        examples_size = len(data)
        if examples_size == 0:
            raise ValueError("the data provided is empty")

        features_size = len(data[0])
        self.examples_size = examples_size
        self.features_size = features_size


        for iteration in range(self.iterations):
            for i in range(0, len(data)):
                example = data[i]
                label = labels[i]
                probability = sigmoid(np.dot(example, self.weights) - self.t)
                for j in range(0, len(example)):
                    self.weights[j] = self.weights[j] - self.learning_rate * (probability - label) * example[j]
                self.t = self.t + self.learning_rate * (probability - label)


