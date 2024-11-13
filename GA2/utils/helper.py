import numpy as np
from mpi4py import MPI
import heapq
import random

class ParallelGraphCentralityCalculator:
    def __init__(self, graph_file, sample_fraction=0.1):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Load the graph data from the file
        self.graph = self.load_graph(graph_file)
        self.n = len(self.graph)
        self.local_n = self.n // self.size
        if self.rank < self.n % self.size:
            self.local_n += 1
            
        self.sample_fraction = sample_fraction

    def load_graph(self, file_path):
        # Determine the graph size by reading the file or set a fixed size
        with open(file_path, 'r') as file:
            max_node = -1
            for line in file:
                u, v = map(int, line.strip().split())
                max_node = max(max_node, u, v)
        
        # Create the graph with appropriate size
        graph = np.zeros((max_node + 1, max_node + 1), dtype=int)
        with open(file_path, 'r') as file:
            for line in file:
                u, v = map(int, line.strip().split())
                graph[u][v] = 1
                graph[v][u] = 1
        return graph
    
    def sample_graph(self, sample_fraction):
        # Get all the edges from the adjacency matrix
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):  # Only consider the upper triangle
                if self.graph[i][j] == 1:
                    edges.append((i, j))

        # Sample 10% of the edges
        sampled_edges = random.sample(edges, int(len(edges) * sample_fraction))

        # Create a new graph matrix with only the sampled edges
        sampled_graph = np.zeros((self.n, self.n), dtype=int)
        for u, v in sampled_edges:
            sampled_graph[u][v] = 1
            sampled_graph[v][u] = 1

        # Update the graph with the sampled edges
        self.graph = sampled_graph

    def calculate_centrality(self):
        local_closeness = np.zeros(self.local_n)
        local_betweenness = np.zeros(self.local_n)

        for i in range(self.rank * self.local_n, (self.rank + 1) * self.local_n):
            local_closeness[i - self.rank * self.local_n] = self.closeness_centrality(i)
            local_betweenness[i - self.rank * self.local_n] = self.betweenness_centrality(i)

        all_closeness = self.comm.gather(local_closeness, root=0)
        all_betweenness = self.comm.gather(local_betweenness, root=0)

        if self.rank == 0:
            closeness = np.concatenate(all_closeness)
            betweenness = np.concatenate(all_betweenness)

            # Print the centrality measures to the "output.txt" file
            with open('output.txt', 'w') as file:
                for i in range(self.n):
                    file.write(f"Node {i}: Closeness Centrality = {closeness[i]:.4f}, Betweenness Centrality = {betweenness[i]:.4f}\n")

            # Print the top 5 nodes with the highest centrality values
            top_closeness = np.argsort(-closeness)[:5]
            top_betweenness = np.argsort(-betweenness)[:5]
            print("Top 5 Nodes by Centrality:")
            print("Closeness Centrality:")
            for node in top_closeness:
                print(f"Node {node}: {closeness[node]:.4f}")
            print("Betweenness Centrality:")
            for node in top_betweenness:
                print(f"Node {node}: {betweenness[node]:.4f}")

            # Print the average of the centrality values
            print(f"Average Closeness Centrality: {np.mean(closeness):.4f}")
            print(f"Average Betweenness Centrality: {np.mean(betweenness):.4f}")

    def closeness_centrality(self, source):
        distances = self.dijkstra(source)
        return 1 / (np.sum(distances) / (self.n - 1))

    def betweenness_centrality(self, source):
        distances = self.dijkstra(source)
        sigma = np.zeros(self.n, dtype=int)
        sigma[source] = 1
        delta = np.zeros(self.n)

        stack = []
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

    def dijkstra(self, source):
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