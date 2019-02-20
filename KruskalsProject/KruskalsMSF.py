import heapq
import re
import sys

class Graph:
    def __init__(self):
        self.heap = []
        self.num = 0
    def addEdge(self,u,v,w):
        heapq.heappush(self.heap, (w,[u,v]))

    def isnotempty (self):
        if len(self.heap) != 0 :
            return True
        else: return False

    def removemin(self):
        return (heapq.heappop(self.heap))

class UnionFind:
    def __init__(self):
        self.parent = []

    def size(self):
        sets = set(self.parent)
        return len(sets)

    def MakeSet(self,x):
        x = self.parent.append(x)

    def find(self,x):
        x0 = self.parent[x]
        while(x0 != self.parent[x0]):
            x0 = self.parent[x0]
        return x0

    def union(self,x,y):
        x0 = self.find(x)
        y0 = self.find(y)
        if (self.parent.count(x0) >= self.parent.count(y0)):
            self.parent = [x0 if i == y0 else i for i in self.parent]
        else:self.union(y,x)

def insert(filename,g):
    with open(filename) as f:
        for i,line in enumerate(f):
            if i == 0:
                g.num = int(line)
            if i >= 1:
                c = re.findall('\d+',line)
                c = [int(v) for i,v in enumerate(c)]
                g.addEdge(c[0],c[1],c[2])

def Kruskals(g,s):
    result = []
    while(len(g.heap) != 0 and s.size() > 1 ):
        e = g.removemin() #e = [weight,[vi,vj]]
        if s.find(e[1][0]) != s.find(e[1][1]):
            #print("add edge (v%d,v%d) = %d to MST"%(e[1][0],e[1][1],e[0]))
            result.append("%d, %d:%d\n"%(e[1][0],e[1][1],e[0]))
            s.union(e[1][0],e[1][1])
    for i in result:
        print(i,end='')
    return result

def check(result,filename):
    sfilename = filename.replace(".",".solution.")
    solution = []
    try:
        with open(sfilename) as f:
            for line in f:
                solution.append(line)
            compare = [i for i, j in zip(result, solution) if i != j]
            if compare == []:
                print("We get the correct solution!")
    except Exception as e:
        print(e)
        print("Could not check correctness without solution!")



g = Graph()
filename = sys.argv[1]
insert(filename,g)

s = UnionFind()
for i in range(g.num):
    s.MakeSet(i)

mst = Kruskals(g,s)

check(mst,filename)
