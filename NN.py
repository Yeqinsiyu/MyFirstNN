import numpy as np
import dataC


SHAPE_OF_LAYERS = [2, 4, 3, 5, 2]
class layer():

    def __init__(self, numOfInputs, numOfNeurons):
        self.weightM = np.random.randn(numOfInputs, numOfNeurons)
        self.biasM = np.random.rand(numOfNeurons)

    def layerForward(self, inputs):
        output = np.dot(inputs, self.weightM) + self.biasM
        normaloutput = dataC.normalizeData(output)
        activeoutput = dataC.activationRelu(normaloutput)
        return activeoutput
    


class network():
    def __init__(self, shapeOfLayers):
        self.layers = []
        for i in range(len(shapeOfLayers) - 1):
            Layer = layer(shapeOfLayers[i], shapeOfLayers[i + 1])
            self.layers.append(Layer)


    def netForward(self, inputs):
        self.outputs = [inputs]

