import dataC
from NN import layer


def main():
    data = dataC.createData(10)
    notagdata = dataC.delTagData(data)
    print('data:', notagdata, '\n')
    Layer1 = layer(2 ,3)
    print(Layer1.layerForward(notagdata))
    

main()