#!/usr/bin/env python
#
# It's to find out max flow in a flow network
# It's similar to Ford-Fulkerson algorithm,
# but defines search order for augmenting path
# The path found must be a shortest path that has available capacity.
# http://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm
#


class Graph:

    def __init__(self):
        self.node = []
        self.edge = {}
        self.neighbors = {}
        self.graph = []  # residual graph
        self.residual = []
        self.row = None
        self.source = None
        self.sink = None
        self.sset = None
        self.tset = None

    # add nodes
    def add_node(self, node=[]):
        self.node = node

    # add edges
    def add_edge(self, node=(), capacity=None):
        self.edge.setdefault(node, capacity)

    # build the network flow
    def build_flow(self, source=None, sink=None):
        for i in range(len(self.node)):
            self.graph.append([])
            self.graph[i] = [0 for j in range(len(self.node))]
            self.neighbors.setdefault(i, [])
        for i, j in self.edge.keys():
            self.graph[i][j] = self.edge[(i, j)]
            self.neighbors[i].append(j)
            if i not in self.neighbors[j]:
                self.neighbors[j].append(i)
        self.residual = [i[:] for i in self.graph]
        self.row = len(self.graph)
        self.source = source
        self.sink = sink

    def edmonds_karp(self):
        flow = 0
        length = len(self.graph)
        flows = [[0 for i in range(length)] for j in range(length)]
        while True:
            max, parent = self.bfs(flows)
            print(max)
            if max == 0:
                self.sset = [self.source] + \
                    [i for i, v in enumerate(parent) if v >= 0]
                self.tset = [x for x in self.node if x not in self.sset]
                print(self.sset, self.tset)
                break
            flow = flow + max
            v = self.sink
            while v != self.source:
                u = parent[v]
                flows[u][v] = flows[u][v] + max
                self.residual[u][v] -= max
                flows[v][u] = flows[v][u] - max
                self.residual[v][u] += max
                v = u
        return flow, flows

    def bfs(self, flows):
        length = self.row
        parents = [-1 for i in range(length)]  # parent table
        parents[self.source] = -2  # make sure source is not rediscovered
        M = [0 for i in range(length)]  # Capacity of path to vertex i
        M[self.source] = float('Inf')  # this is necessary!

        queue = []
        queue.append(self.source)
        while queue:
            u = queue.pop(0)
            for v in self.neighbors[u]:
                # if there is available capacity and v is is not seen before in
                # search
                if self.graph[u][v] - flows[u][v] > 0 and parents[v] == -1:
                    parents[v] = u
                    # it will work because at the beginning M[u] is Infinity
                    M[v] = min(M[u], self.graph[u][v] - flows[u]
                               [v])  # try to get smallest
                    if v != self.sink:
                        queue.append(v)
                    else:
                        return M[self.sink], parents
        return 0, parents

    def find_cut(self):
        cut = {}
        for i in self.sset:
            cut[i] = 0
        for i in self.tset:
            cut[i] = 1
        return cut


def test():
    g = Graph()
    g.add_node(list(range(7)))
    g.add_edge((0, 1), 3)
    g.add_edge((0, 3), 3)
    g.add_edge((1, 2), 4)
    g.add_edge((2, 0), 3)
    g.add_edge((2, 3), 1)
    g.add_edge((2, 4), 2)
    g.add_edge((3, 4), 2)
    g.add_edge((3, 5), 6)
    g.add_edge((4, 1), 1)
    g.add_edge((4, 6), 1)
    g.add_edge((5, 6), 9)
    g.build_flow(0, 6)
    for line in g.graph:
        print(line)
    print(g.neighbors)
    flow, flows = g.edmonds_karp()
    print('Max flow:', flow)
    cut = g.find_cut()
    print('Min cut:', cut)
