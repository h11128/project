import numpy as np
import matplotlib.pyplot as plt


# convert a attribute file into attribute list: [[typename],[type1,type2,type3],[1,2,3]]
attribute = []
attribute.append([["x0"],[0],[1]])
print("attribute list")
with open('game_attributes.txt', 'r', encoding='utf-8') as attributes:
    for line in attributes:
        line = [x.strip("\n").split(",") for x in line.split(":")]
        line[1] = [x.strip(" ") for x in line[1]]
        line.append([x for x in range(len(line[1]))])
        print(line)
        attribute.append(line)

# vectorize file into vector with one-hot
def vectorize1(filename):
    x = []
    y = []
    with open(filename, 'r', encoding='utf-8') as data:
        for line in data:
            line = [x.strip() for x in line.split(',')]
            vector = [1]
            for i in range(len(line)):
                index = attribute[i+1][1].index(line[i])
                if i <= 3:
                    value = [0,0,0]
                    value[index] = 1
                    vector+= value
                elif len(line)-1>i>3 :
                    value = attribute[i+1][2][index]
                    vector.append(value)
                else:
                    y.append(index)
            x.append(vector)
        return x,y

# vectorize file into vector without one-hot
def vectorize2(filename):
    x = []
    y = []
    y.append(1)
    x.append([1,1,1,1,1,1,1,1])
    with open(filename, 'r', encoding='utf-8') as data:
        for line in data:
            line = [x.strip() for x in line.split(',')]
            vector = []
            for i in range(len(line)):

                index = attribute[i+1][1].index(line[i])
                value = attribute[i+1][2][index]
                if i <len(line)-1:
                    vector.append(value)
                else:
                    y.append(index)
            x.append(vector)
        return x,y

x_train, y_train = vectorize1('game_attrdata_train.dat')
x_test, y_test = vectorize1('game_attrdata_test.dat')

def h(x,w):
    if np.dot(w,x)>= 0:
        return 1
    else:
        return 0

def accuracy(w,x,y):
    true = 0
    accuracy = 0
    for i in range(len(y)):
        if h(x[i],w) == y[i]:
            true+= 1
    accuracy = true/len(x)
    return accuracy

# acc is accuracy list[[],[],[],[]]. 0 is current model on train, 1 is average model on train,
# 2 is current model on test, 3 is average model on test
def iteration(x_train,x_test,iter,w=np.zeros([len(x_train[0])])):
    acc = [[],[],[],[]]
    # trainning iteration
    t = np.zeros([len(x_train[0])])
    ta = np.zeros([len(x_train[0])])
    x = []
    for k in range(iter):
        for i in range(len(x_train)):
            change = y_train[i]-h(x_train[i],w)
            w = w+ np.multiply(x_train[i], change)
            t = t + w
        if (k+1)% 1 == 0:
            ta = np.multiply(t, 1/((k+1)*(i+1)))
            x.append(k+1)
            acc[0].append(accuracy(w,x_train,y_train))
            acc[1].append(accuracy(ta,x_train,y_train))
            acc[2].append(accuracy(w,x_test,y_test))
            acc[3].append(accuracy(ta,x_test,y_test))
    return acc,w,ta,x
acc,w,ta,x = iteration(x_train,x_test,250)

def plotaccuracy(acc,title,x):
    plt.plot(x,acc[0],label = "cur on train")
    plt.plot(x,acc[1],label = "ave on train")
    plt.plot(x,acc[2],label = "cur on test")
    plt.plot(x,acc[3],label = "ave on test")
    plt.legend()
    plt.ylabel("accuracy")
    plt.xlabel("number of training epoch")
    plt.title(title)
    plt.show()
plotaccuracy(acc,"prediction",x)

#Q_b
print("""
b)
Average model works better than current model both on train set and test set
""")

def Q_c(x_train,x_test,acc,attribute,x):
    ymax = max(acc[3])
    pos = acc[3].index(ymax)
    xmax = x[pos]
    acc,w,ta,x = iteration(x_train,x_test,xmax)
    print("c)\n the best model is\n",ta)
    math_d = ""
    wmax = 0
    max_index = 0
    X = ["x0",'Weekday', 'Saturday', 'Sunday', 'morning', 'afternoon', 'evening','<30', '30-60', '>60', 'silly', 'happy', 'tired', 'friendsVisiting','kidsPlaying', 'atHome', 'snacks']
    for i in range(len(ta)):
        if i ==0:
            pos = i
        elif 11>i >0:
            pos = int((i+1)/3)
        else:
            pos = int((i-7))
        if abs(ta[i])>wmax:
            max_index=pos
            wmax = ta[i]
        if i==0 or ta[i]<0:
            math_d += str(ta[i])+"(%s) "%(X[i])
        elif ta[i] >=0:
            math_d += "+"+str(ta[i])+"(%s) "%(X[i])
    print(
"""
so the math description of decision function is w·X = %s
where threshold is 0, if bigger than threshold then predict value is SettersOfCatan, and otherwise
X = ["x0",'Weekday', 'Saturday', 'Sunday', 'morning', 'afternoon', 'evening','<30', '30-60', '>60', 'silly', 'happy', 'tired', 'friendsVisiting','kidsPlaying', 'atHome', 'snacks']
the attribute %s plays the most important role that have coeffecient %d
    """%(math_d,attribute[max_index][0][0],wmax)
)

Q_c(x_train,x_test,acc,attribute,x)

def subplotaccuracy(acc,axes,fig,i,title,xx):
    x = int(i/2)
    y = i-2*int(i/2)
    #axes[x, y].plot(xx,acc[0],label = "cur on train")
    #axes[x, y].plot(xx,acc[1],label = "ave on train")
    #axes[x, y].plot(xx,acc[2],label = "cur on test")
    axes[x, y].plot(xx,acc[3],label = "ave on test")
    #axes[int(i/2), i-2*int(i/2)].legend()
    axes[x, y].set_ylabel("accuracy")
    axes[x, y].set_xlabel("number of training epoch")
    axes[x, y].set_title(title)

# ablation test
def ablation(x_train,x_test,isprint):
    if isprint:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 8))
        print("d)")
    min =100
    amin = 9
    for i in range(8):
        acc = [[],[],[],[]]

        if i*3 <12:
            x_trim1 = [x[0:(i)*3+1]+x[(i+1)*3+1:] for x in x_train]
            x_trim2 = [x[0:(i)*3+1]+x[(i+1)*3+1:] for x in x_test]
        elif i <=7:
            x_trim1 = [x[0:9+i]+x[10+i:] for x in x_train]
            x_trim2 = [x[0:9+i]+x[10+i:] for x in x_test]
        else:
            x_trim1 = [x[0:9+i]for x in x_train]
            x_trim2 = [x[0:9+i]for x in x_train]
        w = np.zeros([len(x_trim1[0])])
        t = np.zeros([len(x_trim2[0])])
        acc,w,ta,xx = iteration(x_trim1,x_trim2,250,np.zeros([len(x_trim1[0])]))
        if acc[3][-1]<min:
            min = acc[3][-1]
            att = i+1
        if isprint:
            print("without attribute",attribute[i+1][0],"accuracy:",acc[3][-1])
            subplotaccuracy(acc,axes,fig,i,"Give out %s %s"%("attribute",attribute[i+1][0][0]),xx)

    if isprint:
        print("the most important attribute according to ablation is",attribute[att][0])
        fig.tight_layout()
        plt.show()
    else:
        return attribute[att][0]

ablation(x_train,x_test,True)

"""
#experiment on the question e. With w set to be number randomly sample from N(0,1)
# the result shows weights method is more stable
def Q_e(x_train,x_test,attribute,times):
    a1 = []
    a2 = []
    for j in range(times):
        acc,w,ta,x = iteration(x_train,x_test,250,np.random.randn(len(x_train[0])))
        wmax = 0
        max_index = 0
        X = ["x0",'Weekday', 'Saturday', 'Sunday', 'morning', 'afternoon', 'evening','<30', '30-60', '>60', 'silly', 'happy', 'tired', 'friendsVisiting','kidsPlaying', 'atHome', 'snacks']
        for i in range(len(ta)):
            if i ==0:
                pos = i
            elif 11>i >0:
                pos = int((i+1)/3)
            else:
                pos = int((i-7))
            if abs(ta[i])>wmax:
                max_index=pos
                wmax = ta[i]
        a1.append(attribute[max_index][0])
        a2.append(ablation(x_train,x_test,False))
    print(a1)
    print(a2)
    return a1,a2

a1,a2 = Q_e(x_train,x_test,attribute,50)
"""
print("""

e) No matter how many experiment. The weights method have a stable result which is all mood. The ablation test method will sometimes produce different result. Also if the weights are close on the influential attributes, there would be equally most important attribute. In the ablation test method with different attributes the models are totally different. Simply compare their accuracy is not fair and effective.While on the weights method the comparison is under the same model which is more fair and accurate. So examine the weights is better.
f) The averaged model should be better because it generalize better to test data than the final train model.
""")
