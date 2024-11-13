import numpy as np
from mpi4py import MPI
import heapq
import matplotlib.pyplot as plt
import networkx as nx

class ParallelGraphCentralityCalculator:
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.local_n = self.n // self.size
        if self.rank < self.n % self.size:
            self.local_n += 1

    def calculate_closeness_centrality(self):
        """
        Calculate closeness centrality for each node in the graph using a parallel algorithm.
        
        Returns:
        numpy.ndarray: Array of closeness centrality values for each node.
        """
        local_distances = np.zeros((self.local_n, self.n))
        
        # Calculate distances from each local node to all other nodes
        for i in range(self.rank * self.local_n, (self.rank + 1) * self.local_n):
            local_distances[i - self.rank * self.local_n] = self.dijkstra(i)
        
        # Gather all local distance arrays to the root process
        all_distances = self.comm.gather(local_distances, root=0)
        
        if self.rank == 0:
            # Compute closeness centrality on the root process
            all_distances = np.vstack(all_distances)
            closeness = 1 / (all_distances.sum(axis=1) / (self.n - 1))
            return closeness
        else:
            return None

    def calculate_betweenness_centrality(self):
        """
        Calculate betweenness centrality for each node in the graph using a parallel algorithm.
        
        Returns:
        numpy.ndarray: Array of betweenness centrality values for each node.
        """
        local_betweenness = np.zeros(self.local_n)
        
        # Calculate betweenness centrality for each local node
        for i in range(self.rank * self.local_n, (self.rank + 1) * self.local_n):
            local_betweenness[i - self.rank * self.local_n] = self.brandes(i)
        
        # Gather all local betweenness arrays to the root process
        all_betweenness = self.comm.gather(local_betweenness, root=0)
        
        if self.rank == 0:
            # Compute total betweenness centrality on the root process
            all_betweenness = np.concatenate(all_betweenness)
            return all_betweenness
        else:
            return None

    def dijkstra(self, source):
        """
        Calculate the distances from a given source node to all other nodes in the graph using Dijkstra's algorithm.
        
        Parameters:
        source (int): The index of the source node.
        
        Returns:
        numpy.ndarray: Array of distances from the source node to all other nodes.
        """
        distances = np.full(self.n, np.inf)
        distances[source] = 0
        
        pq = [(0, source)]
        while pq:
            dist, node = heapq.heappop(pq)
            if dist > distances[node]:
                continue
            
            for neighbor, weight in enumerate(self.graph[node]):
                if weight > 0:
                    new_dist = dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return distances

    def brandes(self, source):
        """
        Calculate the betweenness centrality of a given source node in the graph using Brandes' algorithm.
        
        Parameters:
        source (int): The index of the source node.
        
        Returns:
        float: The betweenness centrality of the source node.
        """
        distances = self.dijkstra(source)
        
        stack = []
        sigma = np.zeros(self.n, dtype=int)
        sigma[source] = 1
        delta = np.zeros(self.n)
        
        for i in range(self.n):
            if i != source:
                if distances[i] != np.inf:
                    for j in range(i):
                        if self.graph[j][i] > 0 and distances[j] + self.graph[j][i] == distances[i]:
                            sigma[i] += sigma[j]
                stack.append(i)
        
        while stack:
            w = stack.pop()
            for v in range(self.n):
                if self.graph[w][v] > 0 and distances[v] == distances[w] + self.graph[w][v]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != source:
                betweenness = delta[w]
        
        return betweenness
    
    def visualize_graph(self):
        """
        Visualize the graph using Matplotlib and NetworkX.
        """
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i, self.n):
                if self.graph[i][j] > 0:
                    G.add_edge(i, j, weight=self.graph[i][j])

        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', width=2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
        plt.title('Graph Visualization')
        plt.show()

        