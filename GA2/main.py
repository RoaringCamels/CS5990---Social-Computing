from utils.helper import ParallelClosenessCentralityCalculator
from utils.utils import Utilities
import numpy as np
from mpi4py import MPI
import networkx as nx

def read_graph(file_path, percentage=1.0):
    """
    Read graph from file and return NetworkX graph object
    
    Parameters:
    -----------
    file_path : str
        Path to the graph file
    percentage : float
        Percentage of data to read (between 0 and 1)
        Default is 1.0 (100% of data)
    
    Returns:
    --------
    NetworkX graph object
    """
    if not 0 < percentage <= 1:
        raise ValueError("Percentage must be between 0 and 1")
    
    # First count total number of valid lines
    total_lines = 0
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):  # Skip comments
                total_lines += 1
    
    # Calculate how many lines to read
    lines_to_read = int(total_lines * percentage)
    
    # Read the specified percentage of lines
    G = nx.Graph()
    current_line = 0
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):  # Skip comments
                if current_line < lines_to_read:
                    source, target = map(int, line.strip().split())
                    G.add_edge(source, target)
                    current_line += 1
                else:
                    break
    
    print(f"Read {current_line} edges out of {total_lines} total edges ({percentage*100:.1f}%)")
    print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    return G

def test():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Read graph (only root process needs to read)
    if rank == 0:
        print(f"Process {rank}: Reading graph...")
        
    # Set the path to your data file
    data_path = "GA2/data/facebook_combined.txt"
    
    # Read the graph
    graph = read_graph(data_path, 0.05)
    
    # Create calculator instance with the graph
    calculator = ParallelClosenessCentralityCalculator(graph)
    
    # Calculate centrality
    centrality, execution_time = calculator.calculate_closeness_centrality()
    
    # Only root process prints the timing
    if rank == 0 and execution_time is not None:
        print(f"\nExecution time: {execution_time:.2f} seconds")
        print(f"Number of processors used: {size}")
"""
Run this code with mpirun and choose number of nodes

On mac mpirun -n 4 python3 GA2/main.py
On Windows mpiexec -n 4 python GA2/main.py
"""
if __name__ == "__main__":
    test()