import random

from networkx import Graph
import networkx as nx
def random_walk(graph, start_node, walk_length=100):
        """
        Perform a random walk starting from the given node for a specified number of steps.
        Returns the final node reached after the random walk.
        """
        current_node = start_node
        for _ in range(walk_length):
            neighbors = list(graph.neighbors(current_node))
            if not neighbors:  # Avoid hanging on isolated nodes
                break
            current_node = random.choice(neighbors)
        return current_node

class NetworkHelper:
    def __init__(self):
        pass

    def largestConnectedComponent(self, network: Graph) -> Graph:
        Gcc = max(nx.connected_components(network), key=len)
        return network.subgraph(Gcc)

    def averageDegrees(self, graph:Graph) -> float:
        return sum(dict(graph.degree()).values()) / len(graph)

    def averageShortestPathLength(self, graph: Graph) -> float:
        return nx.average_shortest_path_length(graph)

    def averageClusteringCoefficient(self, graph:Graph) -> float:
        return nx.average_clustering(graph)
    
    def getSize(self, graph: Graph) -> int:
        return graph.number_of_nodes()
    
    
    def averagePathLengthByPercentage(self, G, sample_percentage=0.1):
        # Calculate the sample size based on the percentage
        sample_size = int(len(G) * sample_percentage)
        nodes = list(G.nodes())
        sample_nodes = random.sample(nodes, sample_size)
        
        # Create a subgraph induced by the sampled nodes
        subgraph = G.subgraph(sample_nodes).copy()
        
        # Ensure the subgraph is connected by selecting the largest connected component
        if not nx.is_connected(subgraph):
            largest_component = max(nx.connected_components(subgraph), key=len)
            subgraph = subgraph.subgraph(largest_component).copy()
        
        # Calculate the average shortest path length of the largest connected component
        avg_path_length = nx.average_shortest_path_length(subgraph)
        
        return avg_path_length

    def averagePathLengthRandomWalk(self, graph, num_walks=10000, walk_length=100, centrality=False):
        total_distance = 0
        walk_count = 0
        
        # Sample nodes: either by centrality or uniform sampling
        if centrality:
            # Use degree centrality as an example to weight nodes by their degree
            centrality_scores = nx.degree_centrality(graph)
            nodes = list(graph.nodes())
            weights = [centrality_scores[node] for node in nodes]
        else:
            nodes = list(graph.nodes())
            weights = [1] * len(nodes)  # Uniform sampling
        
        # Normalize the weights to sum to 1 (for probabilistic sampling)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        for _ in range(num_walks):
            # Sample start node based on the computed weights
            start_node = random.choices(nodes, normalized_weights, k=1)[0]
            end_node = random_walk(graph, start_node, walk_length)
            
            # Calculate the path length by computing the distance between start and end nodes
            if nx.has_path(graph, start_node, end_node):
                path_length = nx.shortest_path_length(graph, source=start_node, target=end_node)
                total_distance += path_length
                walk_count += 1
        
        if walk_count == 0:
            return float('inf')  # Avoid division by zero if no valid walks were found
        
        # Return the estimated average path length
        return total_distance / walk_count
    
    def watts_strogatz_graph(self, n, k, p=0.1):
        G = nx.Graph()
        nodes = list(range(n))  # Nodes are labeled 0 to n-1

        # Step 1: Create a regular ring lattice
        for j in range(1, k // 2 + 1):  # k is odd then k-1
            # Targets are the k nearest neighbors
            targets = nodes[j:] + nodes[:j]
            G.add_edges_from(zip(nodes, targets))

        # Step 2: Rewire edges with probability p
        for j in range(1, k // 2 + 1):
            targets = nodes[j:] + nodes[:j]
            for u, v in zip(nodes, targets):
                if random.random() < p:  # With probability p, rewire the edge
                    w = random.choice(nodes)
                    # Avoid self-loops or duplicate edges
                    while w == u or G.has_edge(u, w):
                        w = random.choice(nodes)
                        if G.degree(u) >= n - 1:
                            break  # Skip if the node has all possible edges
                    else:
                        # Rewire the edge (u, v) to (u, w)
                        G.remove_edge(u, v)
                        G.add_edge(u, w)
                        
        return G
    
    def barabasi_albert_graph(self, n, m):
        # Step 1: Initialize the graph with a small initial star graph of m+1 nodes
        G = nx.star_graph(m)

        # Step 2: Create a list of nodes with repetitions based on degree
        repeated_nodes = [node for node, degree in G.degree() for _ in range(degree)]

        # Step 3: Add remaining nodes with preferential attachment
        source = len(G)
        while source < n:
            # Choose m unique nodes from repeated_nodes list based on degree
            targets = set()
            while len(targets) < m:
                selected = random.choice(repeated_nodes)
                targets.add(selected)

            # Add new node `source` and connect it to `m` target nodes
            G.add_edges_from((source, target) for target in targets)

            # Update the repeated nodes list for preferential attachment
            repeated_nodes.extend(targets)
            repeated_nodes.extend([source] * m)

            source += 1  # Move to the next node to add
        
        return G
    

    def watts_strogatz_graph_prebuilt(self, graph: Graph, k) -> Graph:
        return nx.watts_strogatz_graph(len(graph), k , 0.1)
    
    def barabasi_albert_graph_prebuilt(self, graph: Graph, m) -> Graph:
        return nx.barabasi_albert_graph(len(graph), m)