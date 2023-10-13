import numpy as np
import dataC 
import NN


network = NN.network(NN.SHAPE_OF_LAYERS)

for i in range(len(network.layers)):
    print(network.layers[i].weightM, network.layers[i].biasM, '\n')