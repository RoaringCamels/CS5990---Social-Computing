from mpi4py import MPI
import networkx as nx
import time

class ParallelClosenessCentralityCalculator:
    """
    Parallel implementation of closeness centrality calculation.
    
    Space Complexity for the entire class:
    - O(|V| + |E|) for storing the graph
    - O(|V|/P) for local centrality storage per processor
    where |V| is number of vertices, |E| is number of edges, P is number of processors
    """
    
    def __init__(self, graph):
        """
        Initialize calculator with a graph and MPI components
        
        Time Complexity: O(1) - constant time operations
        Space Complexity: O(1) - stores only references and basic variables
        
        Parameters:
            graph: NetworkX graph object
        """
        self.graph = graph
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()    # id of the processor
        self.size = self.comm.Get_size()    # total number of processors available
        self.num_nodes = len(graph)
        
    def calculate_node_assignments(self):
        """
        Distribute nodes among processors for balanced workload
        
        Time Complexity: O(|V|) - to create list of nodes
        Space Complexity: O(|V|/P) - stores subset of nodes for each processor
        where |V| is number of vertices and P is number of processors
        
        Returns:
            List of nodes assigned to current processor
        """
        nodes = list(self.graph.nodes())  # O(|V|)
        nodes_per_proc = len(nodes) // self.size        # each processor is assigned around the same number of nodes
        remainder = len(nodes) % self.size
        
        start_idx = self.rank * nodes_per_proc + min(self.rank, remainder)
        end_idx = start_idx + nodes_per_proc + (1 if self.rank < remainder else 0)
        
        return nodes[start_idx:end_idx]  # O(|V|/P) space
        
    def calculate_closeness_centrality(self):
        """
        Compute closeness centrality in parallel using all available processors
        
        Time Complexity: 
        - O(|V|/P * (|V| + |E|)) for each processor's computation
        - O(|V| log |V|) for sorting in print_analysis
        where |V| is vertices, |E| is edges, P is processors
        
        Space Complexity:
        - O(|V|/P) for local_centrality dictionary per processor
        - O(|V|) for combined results in root processor
        """
        start_time = time.time()
        
        my_nodes = self.calculate_node_assignments()  # O(|V|)
        local_centrality = {}  # O(|V|/P) space
        
        # For each assigned node
        for node in my_nodes:  # O(|V|/P) iterations
            # O(|E|) time for shortest paths using NetworkX
            # Calculate the shortest paths to all other nodes
            length_dict = nx.single_source_shortest_path_length(self.graph, node)
            
            # O(|V|) time to sum distances
            total_distance = sum(length_dict.values())
            if total_distance > 0:
                local_centrality[node] = (len(self.graph) - 1) / total_distance
            else:
                local_centrality[node] = 0.0
        
        # O(|V|) communication time
        all_centrality = self.comm.gather(local_centrality, root=0)
        
        if self.rank == 0:
            # O(|V|) time and space to combine results
            combined_centrality = {}
            for proc_centrality in all_centrality:
                combined_centrality.update(proc_centrality)
            
            self.write_results(combined_centrality)  # O(|V| log |V|)
            self.print_analysis(combined_centrality)  # O(|V| log |V|)
            
            end_time = time.time()
            execution_time = end_time - start_time
            return combined_centrality, execution_time
        
        return None, None
    
    def write_results(self, centrality):
        """
        Write centrality measures to output file
        
        Time Complexity: O(|V| log |V|) - sorting + writing
        Space Complexity: O(|V|) - for sorted items
        
        Parameters:
            centrality: Dictionary of node centrality values
        """
        with open("output.txt", "w") as f:
            # O(|V| log |V|) for sorting
            for node, value in sorted(centrality.items()):
                f.write(f"Node {node}: {value:.6f}\n")
    
    def print_analysis(self, centrality):
        """
        Print top 5 nodes and average centrality
        
        Time Complexity: O(|V| log |V|) - dominated by sorting
        Space Complexity: O(|V|) - for sorted list
        
        Parameters:
            centrality: Dictionary of node centrality values
        """
        # O(|V| log |V|) for sorting
        top_5 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\nTop 5 Nodes by Closeness Centrality:")
        for node, value in top_5:  # O(1) - only 5 iterations
            print(f"Node {node}: {value:.6f}")
        
        # O(|V|) for sum operation
        avg_centrality = sum(centrality.values()) / len(centrality)
        print(f"\nAverage Centrality: {avg_centrality:.6f}")