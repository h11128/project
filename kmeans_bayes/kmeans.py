#!/usr/bin/python

#
# CSE 5522: HW 2 K-Means Implementation
# Author: Eric Fosler-Lussier
#

from numpy import *
from matplotlib.pyplot import *
import json
from numpy.random import *
import numpy
if sys.version_info[0] == 2:
    range = range

colors=['b+','r+','g+','m+','k+','b*','r*','g*','m*','k*','bo','ro','go','mo','ko'];
epsilon=0.0001
block_with_plot=True

def kmeans_iter(curmeans,data):
    meanassign=kmeans_pointassignments(curmeans,data)
    newmeans=curmeans.copy()
    for i in range(0,len(curmeans)):
        assignedpoints=compress(meanassign==i,data,axis=0)
        if len(assignedpoints)==0:
            newmeans[i]=randn(1,data.shape[1])*std(data,axis=0)+mean(data,axis=0)
        else:
            newmeans[i]=average(assignedpoints,axis=0)
    return newmeans

def kmeans_pointassignments(curmeans,data):
    dist=zeros([len(data),len(curmeans)])
    for i in range(0,len(curmeans)):
        dist[:,i]=linalg.norm(data-curmeans[i],axis=1)
    meanassign=argmin(dist,axis=1)

    return meanassign

# runs k-means for the given number of means and data points
def kmeans(nmeans,data):
    means=randn(nmeans,data.shape[1])*std(data,axis=0)+mean(data,axis=0)
    while True:
        newmeans=kmeans_iter(means,data)
        dist=sum(linalg.norm(means-newmeans,axis=0))
        means=newmeans
        if dist<epsilon:
            break
    return newmeans

def plot_assignments(curmeans,data,labels):
    clf()
    curassign=kmeans_pointassignments(curmeans,data)

    for i in range(0,curmeans.shape[0]):
        tp=compress(curassign==i,data,axis=0)
        plot(tp[:,0],tp[:,1],colors[i])
    for ((x,y),lab) in zip(data,labels):
        text(x+.03, y+.03, lab, fontsize=9)
    plot(curmeans[:,0],curmeans[:,1],'c^',markersize=12)
    show(block=block_with_plot)

def count(curmeans,data,traininglabels,kk):
    curassign=kmeans_pointassignments(curmeans,data)
    label = set(traininglabels)
    #compute p(c), where p(c) is p[c]
    p = [0]*len(label)
    # number of points in traininglabels
    n = len(traininglabels)
    uniqu, counts = unique(traininglabels, return_counts=True)
    k = dict(zip(uniqu, counts))
    k1 = {}
    #count and create dictionary for p(V|c)
    for i in range(len(label)):
        p[i] = k[i] / n
        k1[i] = [0 for _ in range(kk)]

    #compute p (V|C), p(Vi|C) is shown below as a dictionary k1,
    #where key is C, p(Vi|C) is k1[C][i]."
    curassign = list(curassign)
    traininglabels = list(traininglabels)
    for j in range(len(label)):
        dd = [i for i,c in enumerate(traininglabels) if c == j]
        length = len(dd)
        for index in dd:
            k1[j][curassign[index]] += 1/length


    return p, k1

def compute_error(testinglabels,testingpoints,traininglabels,kk):

    # step3 predict class use bayes rule
    # kk is the number of k in k-means
    clf()
    curmeans=kmeans(kk,trainingpoints)
    p, k1 = count(curmeans,trainingpoints,traininglabels,kk)

    curassign=kmeans_pointassignments(curmeans,testingpoints)

    k2 = {}
    vv = list(curassign)
    n = len(vv) # length of testing points
    # k2 is dictionary of P(C|Vi) without α where P(C|Vi) = k2[i][c]
    new = [0] *n
    for i in range(kk):
        k2[i] = [0,0,0,0,0,0,0]
    for i in range(kk): # 0 - 9
        for j in range(7): # 0 - 6
            k2[i][j] = k1[j][i]*p[j]
    for i in k2: # predict class of Vi place in last position
        k2[i].append(k2[i].index(max(k2[i])))
    # new is the list predict class of testing points
    for i in range(n):
        new[i] = k2[vv[i]][-1]
    # compute accuracy

    false = 0
    for i in range(n):
        if testinglabels[i] != new[i]:
            false += 1
    error_rate = false / n
    return error_rate

def step3(testinglabels,testingpoints,traininglabels,kk):

    error = []
    for i in range(10):
        error.append(compute_error(testinglabels,testingpoints,traininglabels,kk))
    arr = numpy.array(error)
    print("error mean:",numpy.mean(arr, axis=0))
    print("error std:",numpy.std(arr, axis=0))

def step4(testinglabels,testingpoints,traininglabels):
    klist = [2, 5, 6, 8, 12, 15, 20, 50]
    for i in klist:
        print("when k =",i)
        step3(testinglabels,testingpoints,traininglabels,i)
    print("Conclusion: as the k in k-means increases, the error rate of bayes\
prediction decrease, the prediction's accuracy increase. ")
    print("But this is not definitely \
the case becasue in k-means we choose random points initially, which may result in the \
fact that the final cluster is not so precise as we want.    \
So when the number of k is close error rate may not decrease as k increase")

f=open('training.json','r')
trainingdata=json.load(f)
traininglabels=array(trainingdata['labels'])
trainingpoints=array(trainingdata['points'])
f.close()
f=open('testing.json','r')
testingdata=json.load(f)
testinglabels=array(testingdata['labels'])
testingpoints=array(testingdata['points'])
f.close()


print('step2')
vectormeans=kmeans(10,trainingpoints)
# step 2, p is list of P(c), k1 is dictionary where p(Vi=i|C=c) = k1[c][i]
p,k1 = count(vectormeans,trainingpoints,traininglabels,10)
print("P(c) is :",p)
print("p(Vi|C) is shown below as a matrix, where matrix[i][j] denote p(V=j|C =i).")
for i in k1:
    print(k1[i])

print()
print("step3")
# step 3
step3(testinglabels,testingpoints,traininglabels,10)

print()
print("step4")
# step 3
step4(testinglabels,testingpoints,traininglabels)

#plot_assignments(vectormeans,trainingpoints,traininglabels)
