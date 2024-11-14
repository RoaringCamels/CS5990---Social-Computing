from mpi4py import MPI
import networkx as nx
import time

class ParallelClosenessCentralityCalculator:
    def __init__(self, graph):
        """
        Initialize calculator with a graph and MPI components
        """
        self.graph = graph
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_nodes = len(graph)
        
    def calculate_node_assignments(self):
        """
        Distribute nodes among processors for balanced workload
        """
        nodes = list(self.graph.nodes())
        nodes_per_proc = len(nodes) // self.size
        remainder = len(nodes) % self.size
        
        start_idx = self.rank * nodes_per_proc + min(self.rank, remainder)
        end_idx = start_idx + nodes_per_proc + (1 if self.rank < remainder else 0)
        
        return nodes[start_idx:end_idx]
        
    def calculate_closeness_centrality(self):
        """
        Compute closeness centrality in parallel using all available processors
        """
        start_time = time.time()
        
        # Get node assignments for this processor
        my_nodes = self.calculate_node_assignments()
        local_centrality = {}
        
        # Calculate centrality for assigned nodes
        for node in my_nodes:
            # Use NetworkX's efficient implementation for shortest paths
            length_dict = nx.single_source_shortest_path_length(self.graph, node)
            
            # Calculate closeness centrality
            total_distance = sum(length_dict.values())
            if total_distance > 0:
                local_centrality[node] = (len(self.graph) - 1) / total_distance
            else:
                local_centrality[node] = 0.0
        
        # Gather results from all processors
        all_centrality = self.comm.gather(local_centrality, root=0)
        
        if self.rank == 0:
            # Combine results
            combined_centrality = {}
            for proc_centrality in all_centrality:
                combined_centrality.update(proc_centrality)
            
            # Write results to file
            self.write_results(combined_centrality)
            
            # Print top 5 nodes and average centrality
            self.print_analysis(combined_centrality)
            
            end_time = time.time()
            execution_time = end_time - start_time
            return combined_centrality, execution_time
        
        return None, None
    
    def write_results(self, centrality):
        """
        Write centrality measures to output file
        """
        with open("output.txt", "w") as f:
            for node, value in sorted(centrality.items()):
                f.write(f"Node {node}: {value:.6f}\n")
    
    def print_analysis(self, centrality):
        """
        Print top 5 nodes and average centrality
        """
        # Get top 5 nodes
        top_5 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\nTop 5 Nodes by Closeness Centrality:")
        for node, value in top_5:
            print(f"Node {node}: {value:.6f}")
        
        # Calculate and print average
        avg_centrality = sum(centrality.values()) / len(centrality)
        print(f"\nAverage Centrality: {avg_centrality:.6f}")

    def read_graph(selg, file_path):
        """
        Read graph from file and return NetworkX graph object
        """
        G = nx.Graph()
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith('#'):  # Skip comments
                    source, target = map(int, line.strip().split())
                    G.add_edge(source, target)
        return G
