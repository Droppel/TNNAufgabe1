import numpy as np
import math

from neuralnet import Neuron, Perzeptron

outputLayerSize = 3 # The amount of Neurons in the Outputlayer
inputLayerSize = 5 # The amount of Neurons in the Inputlayer

# This method parses the given Patternfile into a numpy array
def parsePatternFile(filename):
    patterns = np.zeros((32, 8))
    
    with open(filename) as file:
        commentOffset = 0 # We need to offset the commentlines when adding to the array
        for patt, line in enumerate(file):
            if line[0] == "#":
                commentOffset += 1
                continue
            for indx, value in enumerate(line.split()):
                patterns[patt - commentOffset][indx] = value
    return patterns

# This method trains a given Perzeptron on a given amount of patterns, until the error does not change by much
def train(perz, patterns):
    lastError = 0
     # Training Loop
    while True:
        totalError = 0
        # Loop through patterns
        for p in patterns:
            input = p[:5] # First five values in the patternarray are the inputvalues
            teacher = p[5:] # Last three values in the patternarray are the outputvalues

            output, netm = perz.calcNet(input) # Calculate the Perzeptrons output
            for i in range(outputLayerSize): # Calculate the error
                totalError += math.pow(teacher[i] - output[i], 2)
            perz.train(input, netm, output, teacher) # Train the Perzeptron on the teacher values
        print(round(totalError, 3))
        if math.isclose(lastError, totalError, abs_tol=0.0001): # Stop training if the change to our last iteration is small
            perz.storeWeights("weights.txt")
            break
        lastError = totalError

# Calculate the Error of a given perzeptron on given patterns
def calcError(perz, patterns):
     # Training Loop
    totalError = 0
    # Loop through patterns
    for p in patterns:
        input = p[:5]
        teacher = p[5:]

        output, _ = perz.calcNet(input)
        for i in range(outputLayerSize):
            totalError += teacher[i] - output[i]
    return totalError

if __name__ == '__main__':
    # Initialize Outputlayer
    outputLayer = np.empty(outputLayerSize, dtype=Neuron)
    for n in range(outputLayerSize):
        weights = np.random.uniform(low=-0.5, high=0.5, size=inputLayerSize+1)
        outputLayer[n] = Neuron(weights)

    # Initialize Perzeptron with an etha of 0.1    
    perz = Perzeptron(outputLayer, 0.3)
    
    patterns = parsePatternFile("PA-A-train.txt")

    # Train the Perzeptron
    train(perz, patterns)
    
    # The following three lines can be used instead of the training line to test a given file with weights
    #perz.loadWeights("weights.txt")
    #error = calcError(perz, patterns)
    #print(error)