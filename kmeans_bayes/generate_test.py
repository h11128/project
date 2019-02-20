from numpy.random import *
from numpy import *
from matplotlib.pyplot import *
import json
import numpy
if sys.version_info[0] == 2:
    range = xrange
epsilon=0.0001

def generate(dims=2,gaussianwidth=3.2):
    # Global parameters
    nmeans=7

    scale=10
    deviationpct=.1
    ntrainpoints=500
    ntestpoints=120
    colors=['b+','r+','g+','m+','k+','b*','r*']

    block_with_plot=True

    # generate 'num' gaussan means of 'dim' dimensionality
    def gen_means(num,dim):
        return randn(num,dim)*scale

    def gen_distribution(num):
        values=(rand(num)*deviationpct-deviationpct/2)+0.5
        dist=values.copy()/sum(values)

        return dist

    # get means for data classes
    means=gen_means(nmeans,dims)
    # get priors over classes
    dist=gen_distribution(nmeans)
    # train,test labels are sampled from the multinomial distribution described
    # by the class priors
    trainlabels=argmax(multinomial(1,dist,ntrainpoints),axis=1)
    testlabels=argmax(multinomial(1,dist,ntestpoints),axis=1)

    # generate train points by sampling gaussians
    trainpoints=randn(ntrainpoints,dims)*gaussianwidth
    # i'm sure there was a clever way to do this in numpy but I didn't figure
    # it out...  (email me if you know a non-looping way, I'm curious)
    for i in range(0,ntrainpoints):
        trainpoints[i]+=means[trainlabels[i]]

    # generate test points
    testpoints=randn(ntestpoints,dims)*gaussianwidth
    for i in range(0,ntestpoints):
        testpoints[i]+=means[testlabels[i]]

    # this is a quick way to look at the data.  Note that colors needs to be
    # at least as long as the number of classes
    for i in range(0,nmeans):
        tp=compress(trainlabels==i,trainpoints,axis=0)
        plot(tp[:,0],tp[:,1],colors[i])
    #show(block=block_with_plot)
    return testlabels,testpoints,trainlabels,trainpoints
def generate2(gaussianwidth1=6.4,gaussianwidth=3.2):
    # for bonus 1
    nmeans=7
    dims=2
    scale=10
    deviationpct=.1
    ntrainpoints=500
    ntestpoints=120
    colors=['b+','r+','g+','m+','k+','b*','r*']
    g1 = gaussianwidth1
    g2 = gaussianwidth
    block_with_plot=True


    def gen_means(num,dim):
        return randn(num,dim)*scale

    def gen_distribution(num):
        values=(rand(num)*deviationpct-deviationpct/2)+0.5
        dist=values.copy()/sum(values)
        return dist
    # function that generate npoints with asymmetry sampling g1,g2
    def genpoints(npoints,g1,g2):
        #g1 g2 is different so width is different
        a1 = randn(npoints,1)*g1
        a2 = randn(npoints,1)*g2
        a3 = numpy.concatenate((a1,a2),axis=1)
        return a3

    means=gen_means(nmeans,dims)
    dist=gen_distribution(nmeans)
    trainlabels=argmax(multinomial(1,dist,ntrainpoints),axis=1)
    testlabels=argmax(multinomial(1,dist,ntestpoints),axis=1)

    # generate train points by sampling gaussians
    trainpoints=genpoints(ntrainpoints,g1,g2)
    for i in range(0,ntrainpoints):
        trainpoints[i]+=means[trainlabels[i]]

    testpoints=genpoints(ntestpoints,g1,g2)
    for i in range(0,ntestpoints):
        testpoints[i]+=means[testlabels[i]]


    return testlabels,testpoints,trainlabels,trainpoints
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

def count(curmeans,data,trainlabels,kk):
    curassign=kmeans_pointassignments(curmeans,data)
    label = set(trainlabels)
    #compute p(c), where p(c) is p[c]
    p = [0]*len(label)
    # number of points in trainlabels
    n = len(trainlabels)
    uniqu, counts = unique(trainlabels, return_counts=True)
    k = dict(zip(uniqu, counts))
    k1 = {}
    #count and create dictionary for p(V|c)
    for i in range(len(label)):
        p[i] = k[i] / n
        k1[i] = [0 for _ in range(kk)]

    #compute p (V|C), p(Vi|C) is shown below as a dictionary k1,
    #where key is C, p(Vi|C) is k1[C][i]."
    curassign = list(curassign)
    trainlabels = list(trainlabels)
    for j in range(len(label)):
        dd = [i for i,c in enumerate(trainlabels) if c == j]
        length = len(dd)
        for index in dd:
            k1[j][curassign[index]] += 1/length


    return p, k1

def compute_error(testlabels,testpoints,trainlabels,trainpoints,kk):

    # step3 predict class use bayes rule
    # kk is the number of k in k-means
    clf()
    curmeans=kmeans(kk,trainpoints)
    p, k1 = count(curmeans,trainpoints,trainlabels,kk)

    curassign=kmeans_pointassignments(curmeans,testpoints)

    k2 = {}
    vv = list(curassign)
    n = len(vv) # length of test points
    # k2 is dictionary of P(C|Vi) without α where P(C|Vi) = k2[i][c]
    new = [0] *n
    for i in range(kk):
        k2[i] = [0,0,0,0,0,0,0]
    for i in k2: # 0 - 9
        for j in k1: # 0 - 6
            k2[i][j] = k1[j][i]*p[j]
    for i in k2: # predict class of Vi place in last position
        k2[i].append(k2[i].index(max(k2[i])))
    # new is the list predict class of test points
    for i in range(n):
        new[i] = k2[vv[i]][-1]
    # compute accuracy

    false = 0
    for i in range(n):
        if testlabels[i] != new[i]:
            false += 1
    error_rate = false / n
    return error_rate

def step5():
    print("step5")
    widthlist = [1.2 for i in range(10)]
    for i in range(len(widthlist)):
        widthlist[i] += i
        testlabels,testpoints,trainlabels,trainpoints = generate(2,widthlist[i])
        error = []
        for j in range(10):
            error.append(compute_error(testlabels,testpoints,trainlabels,trainpoints,10))
        arr = numpy.array(error)
        print("when width is", widthlist[i],", error mean is",numpy.mean(arr, axis=0))
    print("Generally speaking, when the invariance of Gassian increase, the error_rate increase")
    print()

def bonus1():
    # here the g2 is fixed 3.2, g1 is changing
    print("bonus1")
    g1 = [1.2 for i in range(10)]
    for i in range(len(g1)):
        g1[i] += i**2
        testlabels,testpoints,trainlabels,trainpoints = generate2(g1[i],3.2)
        error = []
        for j in range(10):
            error.append(compute_error(testlabels,testpoints,trainlabels,trainpoints,10))
        arr = numpy.array(error)
        print("when g2 is 3.2, g1 is", g1[i],", error mean is",numpy.mean(arr, axis=0))
    print("Generally speaking, when the invariance of one dimension of Gassian increase in a large scale,\
the error_rate increase")
    print()

def bonus2():
    print("bonu2")
    dim = [2,3,4,5]
    for i in range(len(dim)):
        testlabels,testpoints,trainlabels,trainpoints = generate(dim[i],3.2)
        error = []
        for j in range(10):
            error.append(compute_error(testlabels,testpoints,trainlabels,trainpoints,10))
        arr = numpy.array(error)
        print("when dim is", dim[i],", error mean is",numpy.mean(arr, axis=0))
    print("Generally speaking, when the dimensionality of Gassian increase, the error_rate decrease rapidly")
#generate(dims=2,gaussianwidth=3.2)
step5()
bonus1()
bonus2()
