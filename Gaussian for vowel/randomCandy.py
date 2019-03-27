import random
import numpy as np
import matplotlib.pyplot as plt
"""
h1: 100% cherry,
h2: 75% cherry + 25% lime,
h3: 50% cherry + 50% lime,
h4: 25% cherry + 75% lime,
h5: 100% lime
The Posterior probablity is calculated by P(hi | D) = αP(D | hi)P(hi)
The prior distribution over h1, . . . , h5 is given by <0.1, 0.2, 0.4, 0.2, 0.1>
The likelihood of the data is calculated under the assumption that the observations
    are i.i.d. : P(D | hi) = P(d1 | hi)P(d2 | hi)....P(dn | hi)
The Bayesian Prediction:
    P(dN+1 = lime | d1,..., dN) =P(dN+1 | D, hi)P(hi | D) = P(dN+1 | hi)P(hi | D)

lime 1 cherry 0
"""
h =[[],[]]
h[1] = [0, 0.25, 0.5, 0.75, 1]
h[0] = [1, 0.75, 0.5, 0.25, 0]

def learn(pp,h):
    D = []
    i = 0

    phi = {}
    phi[0]=[0.1]
    phi[0.25] = [0.2]
    phi[0.5] = [0.4]
    phi[0.75] = [0.2]
    phi[1] = [0.1]
    prediction = []

    while i<100:
        p = random.random()
        if p >=pp:
            D.append(1)
        else: D.append(0)
        i+=1

    likelihood = [1,1,1,1,1]
    post = [0.1,0.2,0.4,0.2,0.1]

    for j in range(100):
        predict = 0
        for i in range(5):
            predict += h[1][i]*phi[h[1][i]][j]
            likelihood[i] *= h[D[j]][i]
            post[i] = likelihood[i]*phi[h[1][i]][0]
        norm = [i/sum(post) for i in post]
        prediction.append(predict)
        for i in range(5):
            phi[h[1][i]].append(norm[i])
    return phi, prediction

def plotpost(phi):
    x = np.arange(0,110,10)
    for i in range(5):
        phi[h[1][i]] = [phi[h[1][i]][k] for k in range(0,110,10)]
        plt.plot(x,phi[h[1][i]],label = "p(h%d|D)"%(i+1))
    plt.legend()
    plt.ylabel("Posterior probablity of hypothesis")
    plt.xlabel("number of samples in D")
    plt.show()

def plotprediction(prediction):

    x = np.arange(0,100,10)
    prediction = [prediction[k] for k in range(0,100,10)]
    plt.ylabel("Probability that next candy is lime")
    plt.xlabel("number of samples in D")
    plt.ylim(0,1)
    plt.plot(x, prediction)
    plt.show()

phi,prediction = learn(0.25,h)
plotpost(phi)
plotprediction(prediction)
phi,prediction = learn(0.5,h)
plotpost(phi)
plotprediction(prediction)
