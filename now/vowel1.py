import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture

data = []
vowel = []
Gau = []
with open('train.txt', 'r', encoding='utf-8') as traindata:
    for line in traindata:
        line = [x.strip('\n') for x in line.split(" ")]
        if line[2] not in vowel:
            Gau.append([line[2]])
            vowel.append(line[2])
        data.append([[float(line[0]),float(line[1])],line[2]])


iprior = [random.random() for _ in range(len(Gau))]
iprior = [i/sum(iprior) for i in iprior]
Imean = [500, 2500]
Icov = [[14400, 0], [0, 62500]]
for i in range(len(Gau)):
    Gau[i].append(Imean)
    Gau[i].append(Icov)
    Gau[i].append(iprior[i])

def partA(Gau,data):
    X = []
    Y = []
    for i in range(len(data)):
        X.append(data[i][0])
        Y.append(vowel.index(data[i][1]))
    X = np.array(X)
    Y = np.array(Y)
    clf = GaussianNB()
    clf.fit(X, Y)
    test = []
    result = []
    true = 0
    with open('test.txt', 'r', encoding='utf-8') as testdata:
        for line in testdata:
            line = [x.strip('\n') for x in line.split(" ")]
            test.append([float(line[0]),float(line[1])])
            result.append(vowel.index(line[2]))
    for i in range(len(test)):
        if clf.predict([test[i]]) == result[i]:
            true+=1
    accuracy = true/len(test)
    for i in range(len(Gau)):
        Gau[i][1] = clf.theta_[i]
        Gau[i][2][0][0] = clf.sigma_[i,0]
        Gau[i][2][1][1] = clf.sigma_[i,1]
    return Gau,accuracy

def partA1(Gau,data):
    Post = []
    for j in range(len(data)):
        post = [0 for _ in range(10)]
        index = vowel.index(data[j][1])
        post[index] = 1
        Post.append(post)
    N = np.sum(Post, axis=0)
    Post = np.array(Post)
    Gau = Mstep(Gau,data,Post,N)
    test = []
    result = []
    with open('test.txt', 'r', encoding='utf-8') as testdata:
        for line in testdata:
            line = [x.strip('\n') for x in line.split(" ")]
            test.append([float(line[0]),float(line[1])])
            result.append(vowel.index(line[2]))
    true = 0
    loglikelihood = 0
    for j in range(len(test)):
            post = [0 for _ in range(10)]
            predict = 0
            for i in range(10):
                loglikelihood += multivariate_normal.logpdf(test[j][0], Gau[i][1], Gau[i][2])
                prior = Gau[i][3]
                post[i] = math.log(prior)+loglikelihood
            predict = post.index(max(post))
            if predict == result[j]:
                true+= 1
    accuracy = true/len(test)
    print(Gau)
    print(accuracy)
    return Gau

def Estep(Gau,data):
    loglikelihood = 0
    Post = []
    for j in range(len(data)):
            post = [0 for _ in range(len(Gau))]
            for i in range(len(Gau)):
                likelihood = multivariate_normal.pdf(data[j][0], Gau[i][1], Gau[i][2])

                prior = Gau[i][3]
                post[i] = prior*likelihood
                loglikelihood += np.log(post[i])
            norm = [i/sum(post) for i in post]
            Post.append(norm)
    N = np.sum(Post, axis=0)
    return Post,N,loglikelihood

def Mstep(Gau,data,Post,N):
    for i in range(len(Gau)):
        newMean = np.array([0,0])
        newCov = np.array([[0, 0], [0, 0]])
        for j in range(len(data)):
            weights = np.dot(Post[j,i]/N[i],data[j][0])
            a = np.subtract(data[j][0],Gau[i][1])
            b = a.transpose()
            c = np.multiply(b,a)
            covs = np.dot(Post[j,i]/N[i],c)
            newMean = np.add(newMean,weights)

            newCov = np.add(newCov,covs)
        newCov[0,1] = 0
        newCov[1,0] = 0
        Gau[i][1] = newMean
        Gau[i][2] = newCov
        Gau[i][3] = N[i]/np.sum(N)

    return Gau

def iteration(covtype,Gau,data,k):

    likelihoods = []
    for i in range(k):
        Post,N,loglikelihood = Estep(Gau,data)
        Post = np.array(Post)
        likelihoods.append(loglikelihood)
        Gau = Mstep(Gau,data,Post,N)
    X = []
    Y = []
    for i in range(len(data)):
        X.append(data[i][0])
        Y.append(vowel.index(data[i][1]))
    X = np.array(X)
    Y = np.array(Y)
    clf = GaussianMixture(k*10,covtype)
    clf.fit(X, Y)
    test = []
    result = []
    true = 0
    with open('test.txt', 'r', encoding='utf-8') as testdata:
        for line in testdata:
            line = [x.strip('\n') for x in line.split(" ")]
            test.append([float(line[0]),float(line[1])])
            result.append(vowel.index(line[2]))
    for i in range(len(test)):
        if clf.predict([test[i]]) == result[i]:
            true+=1
    accuracy = true/len(test)
    print("the accuracy for %d mixture per vowel and cov is %s :"%(k,covtype),accuracy)
    return accuracy


iteration("diag",Gau,data,1)
iteration("diag",Gau,data,2)
iteration("diag",Gau,data,3)
iteration("diag",Gau,data,4)
iteration("diag",Gau,data,5)
for i in range(10):

    print("%d th experiment for EC2 and EC3"%i)
    iteration("diag",Gau,data,1)
    iteration("full",Gau,data,1)

    print(" ")
    iteration("diag",Gau,data,2)
    iteration("full",Gau,data,2)


print("(b) the error shows that increase mixture per vowel to 2 does not improve the model")
print("(c) the error shows that increase mixture per vowel does not improve the model, the performance somehow getting worse")
print("(d) the experiment shows improve is not relative to partA")
print("(d) the improve is not relative to partA or PartB, but relative to EC 2 since the accuracy is the same, which means the final model is the same")
