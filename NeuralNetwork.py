import os

import numpy as np
import math as m

import Proccessing


class Layer:
    size = 1
    neuron_list = []
    weights_list = []

    def __init__(self, size, weights):
        self.size = size
        self.neuron_list = [0.0] * size
        self.weights_list = np.array(weights)

    def output(self, input_list, il=False):
        input_list = np.array(input_list)
        if not il:
            for i in range(self.size):
                self.neuron_list[i] = np.dot(input_list, (self.weights_list[i]).transpose())
        else:
            self.neuron_list = input_list

    def activation(self, ol=False):
        for i in range(self.size):
            self.neuron_list[i] = 1 / (1 + m.exp(-self.neuron_list[i]))
        if ol:
            for i in range(self.size):
                self.neuron_list[i] /= sum(self.neuron_list)

    def print_attr(self):
        print("size = ", self.size)
        for i in range(self.size):
            print(self.neuron_list[i])


class NeuralNetwork:
    size = 1
    X_train = list()
    y_train = list()
    layers = list()

    def __init__(self, size):
        self.size = size[0]
        for i in range(1, self.size + 1):
            dum = []
            if i == 1:
                dum = [1] * size[1]
                self.layers.append(Layer(size[i], dum))
            else:
                # dum = np.random.rand(size[i], size[i - 1])
                dum = Proccessing.rand_lists(size[i], size[i - 1])
                self.layers.append(Layer(size[i], dum))

    def fit(self, x_train, y_train, thresh_hold, epochs=1000, eta=1):
        self.X_train = x_train
        self.y_train = y_train
        iteration = 1
        err = 1.0

        while err >= thresh_hold and iteration <= epochs:
            for i in range(len(self.X_train)):
                self.forward_feed(self.X_train[i])
                self.back_prop(self.y_train[i], eta)
                err = self.error(self.layers[-1].neuron_list, self.y_train[i])
            if iteration % 10 == 0:
                print("epochs=", iteration, "error=", err)
            iteration += 1

    def error(self, predicted, y_train):
        error = 0.0
        # print(predicted)
        # print(y_train)
        for j in range(len(y_train)):
            error += (predicted[j] - y_train[j]) ** 2
        return error

    def back_prop(self, target, eta=0.001):
        segma_k = []
        for i in reversed(range(1, len(self.layers))):
            dum1 = []
            if i == len(self.layers) - 1:
                for j in range(len(self.layers[i].neuron_list)):
                    dum = (target[j] - self.layers[i].neuron_list[j]) * self.layers[i].neuron_list[j] * (
                            1 - self.layers[i].neuron_list[j])
                    dum1.append(dum)
                    for k in range(len(self.layers[i - 1].neuron_list)):
                        self.layers[i].weights_list[j][k] += eta * dum * self.layers[i - 1].neuron_list[k]
                segma_k.append(dum1)
            else:
                for j in range(len(self.layers[i].neuron_list)):
                    dum = 0
                    for k in range(len(segma_k[-1])):
                        dum += segma_k[-1][k] * self.layers[i + 1].weights_list[k][j]
                    dum *= self.layers[i].neuron_list[j] * (1 - self.layers[i].neuron_list[j])
                    dum1.append(dum)
                    for k in range(len(self.layers[i - 1].neuron_list)):
                        self.layers[i].weights_list[j][k] += eta * dum * self.layers[i - 1].neuron_list[k]
                segma_k.append(dum1)

    def forward_feed(self, x_train):
        self.layers[0].output(x_train, True)
        for j in range(1, self.size):
            self.layers[j].output(self.layers[j - 1].neuron_list)
            if j == len(self.layers) - 1:
                self.layers[j].activation(True)
            else:
                self.layers[j].activation()

    def predict(self, x_test):
        predictions = list()
        for i in x_test:
            self.forward_feed(i)
            predictions.extend(self.layers[-1].neuron_list)
        predictions = self.calc_labels(predictions, len(x_test))
        return predictions

    def calc_labels(self, predictions, size):
        dum = list()
        predictions = np.array(predictions)
        predictions = predictions.reshape(size, 10)
        for i in predictions:
            dum.extend(np.where(i == np.amax(i)))
        predictions = dum
        return predictions
