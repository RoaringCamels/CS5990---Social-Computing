import networkx as nx
import time

class SequentialClosenessCentralityCalculator:
    """
    Sequential implementation of centrality calculation.
    
    Space Complexity:
    - O(|V| + |E|) for storing the graph
    - O(|V|) for centrality storage
    where |V| is number of vertices, |E| is number of edges
    """
    
    def __init__(self, graph, centrality_type='closeness'):
        """
        Initialize calculator with a graph
        
        Time Complexity: O(1) - constant time operations
        Space Complexity: O(1) - stores only references and basic variables
        
        Parameters:
            graph: NetworkX graph object
            centrality_type: Type of centrality to calculate ('closeness' or 'betweenness')
        """
        self.graph = graph
        self.num_nodes = len(graph)
        self.centrality_type = centrality_type.lower()

        if self.centrality_type not in ['closeness', 'betweenness']:
            raise ValueError("centrality_type must be either 'closeness' or 'betweenness'")
    
    def calculate_closeness_centrality(self):
        """
        Compute closeness centrality sequentially
        
        Time Complexity: 
        - O(|V| * (|V| + |E|)) for computing shortest paths
        - O(|V| log |V|) for sorting in print_analysis
        
        Space Complexity:
        - O(|V|) for centrality dictionary
        """
        start_time = time.time()
        
        centrality = {}
        
        # Compute closeness centrality for each node
        for node in self.graph.nodes():  # O(|V|)
            # Compute shortest paths from current node to all other nodes
            # O(|V| + |E|) time complexity for each node
            length_dict = nx.single_source_shortest_path_length(self.graph, node)
            
            # Compute total distance
            total_distance = sum(length_dict.values())
            
            # Compute centrality
            if total_distance > 0:
                centrality[node] = (len(self.graph) - 1) / total_distance
            else:
                centrality[node] = 0.0
        
        # Write and print results
        self.write_results(centrality)
        self.print_analysis(centrality)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return centrality, execution_time
    
    def calculate_betweenness_centrality(self):
        """
        Compute betweenness centrality sequentially
        
        Time Complexity: 
        - O(|V| * |V| * |E|) for computing betweenness
        - O(|V| log |V|) for sorting in print_analysis
        
        Space Complexity:
        - O(|V|) for centrality dictionary
        """
        start_time = time.time()
        
        # Use NetworkX's built-in betweenness centrality function
        # O(|V| * |V| * |E|) time complexity
        centrality = nx.betweenness_centrality(self.graph)
        
        # Write and print results
        self.write_results(centrality)
        self.print_analysis(centrality)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return centrality, execution_time
    
    def calculate_centrality(self):
        """
        Calculate either closeness or betweenness centrality based on initialization parameter
        """
        if self.centrality_type == 'closeness':
            return self.calculate_closeness_centrality()
        else:
            return self.calculate_betweenness_centrality()
    
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
        
        print("\nTop 5 Nodes by Centrality:")
        for node, value in top_5:  # O(1) - only 5 iterations
            print(f"Node {node}: {value:.6f}")
        
        # O(|V|) for sum operation
        avg_centrality = sum(centrality.values()) / len(centrality)
        print(f"\nAverage Centrality: {avg_centrality:.6f}")

# Example usage:
# import networkx as nx
# graph = nx.erdos_renyi_graph(100, 0.1)  # Example graph
# calculator = SequentialClosenessCentralityCalculator(graph)
# results, time_taken = calculator.calculate_centrality()