import pandas as pd
import networkx as nx

class Utilities:
    def __init__(self):
        pass
    
    def createGraph(self, pathToData: str, percentage: float=100) -> nx.Graph:
        """
        Creates a graph from a CSV file. Optionally, only a percentage of the rows can be used to create the graph.
        
        Parameters:
        - pathDataToData: str, path to the CSV file containing the graph data.
        - percentage: float, the percentage of the rows to sample from the CSV file (default is 100).
        
        Returns:
        - A NetworkX graph.
        """

        df = pd.read_csv(pathToData, header = None, delimiter=r"\s+")

        # Use percentage of data for testing small portions
        if percentage < 100:
            sampleSize = int(len(df) * (percentage / 100))
            df = df.sample(n=sampleSize, random_state=42)

        # Data has only edges and no weights
        graph = nx.Graph()
        for _, row in df.iterrows():
            node1, node2 = row
            graph.add_edge(node1, node2)

        print("Nodes:", graph.nodes())
        print("Edges:", graph.edges(data=True))

        return graph