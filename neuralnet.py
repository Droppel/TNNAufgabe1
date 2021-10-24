import math
from os import write
import numpy as np
from numpy.core.fromnumeric import size

class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.inputSize = size(weights)

    # Calculate this neurons output based on a input
    def calc(self, inputs):
        result = 0
        for i, x in enumerate(inputs):
            result += x * self.weights[i]
        result += self.weights[-1] # result now equals net_m
        return (fermi(result), result)
    
    # Train the weights of this neuron based on the given parametes
    def train(self, input, netm, output, teacher, etha):
        delta = (teacher - output) * fermiDeriv(netm) # calculate delta using the deltarule for Outputneurons
        ethadelta = etha * delta
        for w in range(self.inputSize - 1): # Calculate the change for every weight base on delta and etha
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

    # Calculate the outputs of the Perzeptron given an input
    def calcNet(self, input):
        output = np.empty(self.outputLayerSize)
        netm = np.empty(self.outputLayerSize) # the method also returns net_m to use in the weightchange step
        for i in range(self.outputLayerSize):
            output[i], netm[i] = self.outputNeurons[i].calc(input)
        return output, netm
    
    # Train every Neuron of the Perzeptron
    def train(self, input, netm, output, teacher):
        for i in range(self.outputLayerSize):
            self.outputNeurons[i].train(input, netm[i], output[i], teacher[i], self.etha)
    
    # Store the current weights in a file. Neurons are seperated by new lines and single weights using a " "
    def storeWeights(self, name):
        store = ""
        f = open(name, "w")
        for neuron in self.outputNeurons:
            for weight in neuron.weights:
                store += " " + str(weight)
            store += "\n"
        f.write(store)
        f.close()

    # Load the weights from a file created using the storeWeights() function
    def loadWeights(self, name):
        with open(name) as file:
            for lineindx, line in enumerate(file):
                for indx, value in enumerate(line.split()):
                    self.outputNeurons[lineindx].weights[indx] = value


# The Logistic Function
def fermi(x):
    return 1 / (1 + math.exp(-1 * x))

# The derivative of the Logistic Function
def fermiDeriv(x):
    return fermi(x) * (1 - fermi(x))