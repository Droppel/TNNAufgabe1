import numpy as np

from neuralnet import Neuron, Perzeptron

outputLayerSize = 3
inputLayerSize = 5

def parseFile(filename):
    patterns = np.zeros((32, 8))
    
    with open(filename) as file:
        commentOffset = 0
        for patt, line in enumerate(file):
            if line[0] == "#":
                commentOffset += 1
                continue
            for indx, value in enumerate(line.split()):
                patterns[patt - commentOffset][indx] = value
    return patterns

def train(perz, patterns):
     # Training Loop
    while True:
        totalError = 0
        # Loop through patterns
        for p in patterns:
            input = p[:5]
            teacher = p[5:]

            output, netm = perz.calcNet(input)
            for i in range(outputLayerSize):
                totalError += teacher[i] - output[i]
            perz.train(input, netm, output, p[5:])
        print(round(totalError, 3))
        if round(totalError, 3) == 0:
            perz.storeWeights()
            break

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
    outputLayer = np.empty(outputLayerSize, dtype=Neuron)
    for n in range(outputLayerSize):
        weights = np.random.uniform(low=-0.5, high=0.5, size=inputLayerSize+1)
        outputLayer[n] = Neuron(weights)
    
    perz = Perzeptron(outputLayer, 0.1)
    
    patterns = parseFile("PA-A-train.txt")

    #train(perz, patterns)
    perz.loadWeights("weights.txt")
    error = calcError(perz, patterns)
    print(error)