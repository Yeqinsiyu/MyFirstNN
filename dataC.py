import numpy as np
import random

def tagData(x, y):
    if x ** 2 + y ** 2 < 1:
        return 1
    else:
        return 0
    
def createData(numberOfdata):
    data = []
    for i in range(numberOfdata):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        if tagData(x, y):
            data.append([x, y, 1])
        else:
            data.append([x, y, 0])
    return np.array(data)

def delTagData(data):
    return data[:, (0, 1)]

def normalizeData(data):
    absdata = np.absolute(data)
    sumofdatarow = np.sum(absdata, axis=1, keepdims=True)
    sumofdatarow = np.where(sumofdatarow == 0 , 1 , 1  / sumofdatarow)
    returndata = data * sumofdatarow
    return returndata

def activationRelu(originaloutput):
    originaloutput = np.where(originaloutput > 0, originaloutput, 0)
    return originaloutput