import math
from os import write
import numpy as np
from numpy.core.fromnumeric import size

class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.inputSize = size(weights)

    def calc(self, inputs):
        result = 0
        for i, x in enumerate(inputs):
            result += x * self.weights[i]
        result += self.weights[-1]
        return (fermi(result), result)
    
    def train(self, input, netm, output, teacher, etha):
        delta = (teacher - output) * fermiDeriv(netm)
        ethadelta = etha * delta
        for w in range(self.inputSize - 1):
            change = ethadelta * input[w]
            self.weights[w] += change
        # Treat bias weight
        change = ethadelta # Here input[w] is 1 and thus not needed
        self.weights[-1] += change

class Perzeptron:
    def __init__(self, outputNeurons, etha):
        self.outputNeurons = outputNeurons
        self.outputLayerSize = size(outputNeurons)
        self.etha = etha

    def calcNet(self, input):
        output = np.empty(self.outputLayerSize)
        netm = np.empty(self.outputLayerSize)
        for i in range(self.outputLayerSize):
            output[i], netm[i] = self.outputNeurons[i].calc(input)
        return output, netm
    
    def train(self, input, netm, output, teacher):
        for i in range(self.outputLayerSize):
            self.outputNeurons[i].train(input, netm[i], output[i], teacher[i], self.etha)
    
    def storeWeights(self, name):
        store = ""
        f = open(name, "w")
        for neuron in self.outputNeurons:
            for weight in neuron.weights:
                store += " " + str(weight)
            store += "\n"
        f.write(store)
    
    def loadWeights(self, name):
        with open(name) as file:
            for lineindx, line in enumerate(file):
                for indx, value in enumerate(line.split()):
                    self.outputNeurons[lineindx].weights[indx] = value



def fermi(x):
    return 1 / (1 + math.exp(-1 * x))

def fermiDeriv(x):
    return fermi(x) * (1 - fermi(x))