import numpy as np
from mpi4py import MPI
import heapq

class ParallelClosenessCentralityCalculator:
    def __init__(self, graph):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Store the graph directly
        self.graph = graph
        self.n = len(self.graph)
        self.local_n = self.n // self.size
        if self.rank < self.n % self.size:
            self.local_n += 1

    def calculate_closeness(self):
        # Calculate local closeness centrality values
        local_closeness = np.zeros(self.local_n)
        start_idx = self.rank * self.local_n
        end_idx = min((self.rank + 1) * self.local_n, self.n)
        
        for i in range(start_idx, end_idx):
            local_closeness[i - start_idx] = self.closeness_centrality(i)

        # Gather all results at root processor
        all_closeness = self.comm.gather(local_closeness, root=0)

        # Process and output results at root processor
        if self.rank == 0:
            # Combine all results
            closeness = np.concatenate(all_closeness)

            # Write all centrality measures to output.txt
            with open('output.txt', 'w') as file:
                for i in range(self.n):
                    file.write(f"Node {i}: Closeness Centrality = {closeness[i]:.4f}\n")

            # Print top 5 nodes with highest centrality
            top_indices = np.argsort(-closeness)[:5]
            print("\nTop 5 Nodes by Closeness Centrality:")
            for idx in top_indices:
                print(f"Node {idx}: {closeness[idx]:.4f}")

            # Print average centrality
            avg_centrality = np.mean(closeness)
            print(f"\nAverage Closeness Centrality: {avg_centrality:.4f}")

    def closeness_centrality(self, source):
        distances = self.dijkstra(source)
        # Filter out unreachable nodes (infinite distances)
        reachable_distances = distances[distances != np.inf]
        if len(reachable_distances) > 1:  # If there are reachable nodes besides the source
            return (len(reachable_distances) - 1) / (np.sum(reachable_distances) * (self.n - 1))
        return 0.0  # Return 0 for isolated nodes

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