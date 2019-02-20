import numpy as np
import matplotlib.pyplot as plt


# convert a attribute file into attribute list: [[typename],[type1,type2,type3],[1,2,3]]
attribute = []
with open('game_attributes.txt', 'r', encoding='utf-8') as attributes:
    for line in attributes:

        line = [x.strip("\n").split(",") for x in line.split(":")]
        line[1] = [x.strip(" ") for x in line[1]]
        line.append([x for x in range(len(line[1]))])
        attribute.append(line)
# vectorize file into vector without one-hot
def vectorize(filename):
    x = []
    y = []
    y.append(1)
    x.append([1,1,1,1,1,1,1,1])
    with open(filename, 'r', encoding='utf-8') as data:
        for line in data:
            line = [x.strip() for x in line.split(',')]
            vector = []
            for i in range(len(line)):
                index = attribute[i][1].index(line[i])
                value = attribute[i][2][index]
                if i <len(line)-1:
                    vector.append(value)
                else:
                    y.append(value)
            x.append(vector)
        return x,y

x_train, y_train = vectorize('game_attrdata_train.dat')
x_test, y_test = vectorize('game_attrdata_test.dat')

n = len(x_train)
w = [0 for x in range(len(x_train[0]))]
t = [0 for x in range(len(x_train[0]))]

def h(x,w):
    wx = sum( [w[i]*x[i] for i in range(len(x))] )
    if wx>= 0:
        return 1
    else:
        return 0

def accuracy(w,x,y):
    true = 0
    for i in range(len(x)):
        y_predict = h(x[i],w)
        if y_predict == y[i]:
            true+= 1
    accuracy = true/len(x)
    return accuracy

acc = []
# trainning iteration
for i in range(n):
    d = y_train[i]-h(x_train[i],w)
    change = [x *d for x in x_train[i]]
    w = [w[j]+change[j] for j in range(len(w))]
    t = [(t[j]+w[j])/n for j in range(len(w))]
    acc_i = [accuracy(w,x_train,y_train),accuracy(t,x_train,y_train),accuracy(w,x_test,y_test),accuracy(w,x_test,y_test)]
    acc.append(acc_i)


def plotaccuracy(acc):
    x = np.arange(0,len(acc),1)
    y = np.asarray(acc)
    plt.plot(x,y)
plotaccuracy(acc)
