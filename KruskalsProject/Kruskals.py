from collections import defaultdict
# adjacency list
import heapq
import re
import sys

class Graph:
    def __init__(self):
        self.heap = []
        self.graph = defaultdict(list)

    def addEdge(self,u,v,w):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.num = len(self.graph)
        heapq.heappush(self.heap, (w,[u,v]))

    def connected(self,v,visited,u):
        if (not u in self.graph) or (not v in self.graph):
            return False

        visited[v] = 1
        for i in self.graph[v]:
            if i == u:
                return True
            if visited[i] == 0:
                if self.connected(i,visited,u):
                    return True
        return False

    def addTreeEdge(self,u,v,w):
        visited = [0]*(self.num)
        if self.connected(u,visited,v):
            return False
        else:
            self.graph[u].append(v)
            self.graph[v].append(u)
            return True

    def removemin(self):
        return (heapq.heappop(self.heap))

def insert(filename,g):
    with open(filename) as f:
        for i,line in enumerate(f):
            if i == 0:
                g.num = int(line)
            if i >= 1:
                c = re.findall('\d+',line)
                c = [int(v) for i,v in enumerate(c)]
                g.addEdge(c[0],c[1],c[2])

def Kruskals(g):
    g1 = Graph()
    g1.num = g.num
    result = []
    while(len(g.heap) != 0 and len(g1.graph) < g1.num ):
        e = g.removemin()
        if g1.addTreeEdge(e[1][0],e[1][1],e[0]):
            #print("add edge (v%d,v%d) = %d to MST"%(e[1][0],e[1][1],e[0]))
            result.append("%d, %d:%d\n"%(e[1][0],e[1][1],e[0]))
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



g = Graph()
filename = sys.argv[1]
insert(filename,g)

mst = Kruskals(g)
print(mst)
check(mst,filename)
