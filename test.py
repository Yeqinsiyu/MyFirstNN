import numpy as np
import dataC 
import NN


data = dataC.createData(4)
data = dataC.delTagData(data)
a = np.array([[0, 0]])


print(dataC.normalizeData(a))