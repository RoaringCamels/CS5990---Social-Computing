from utils.helper import ParallelClosenessCentralityCalculator
import numpy as np
import os
from mpi4py import MPI

def count_lines(file_path):
    """Count total lines in the file."""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def read_partial_graph(file_path, num_lines=None, start_line=0):
    """
    Read a portion of the graph file and return the maximum node number
    and the list of edges.
    """
    edges = []
    max_node = -1
    
    with open(file_path, 'r') as file:
        # Skip to start_line
        for _ in range(start_line):
            next(file)
            
        # Read specified number of lines
        count = 0
        for line in file:
            if num_lines is not None and count >= num_lines:
                break
                
            u, v = map(int, line.strip().split())
            edges.append((u, v))
            max_node = max(max_node, u, v)
            count += 1
            
    return max_node, edges

def create_graph(max_node, edges):
    """Create adjacency matrix from edges."""
    size = max_node + 1
    graph = np.zeros((size, size), dtype=int)
    
    for u, v in edges:
        graph[u][v] = 1
        graph[v][u] = 1
        
    return graph

def test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    try:
        data_path = "GA2/data/facebook_combined.txt"
        
        # Check if the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        # Process in chunks of 1000 lines
        chunk_size = 1000
        
        if rank == 0:
            total_lines = count_lines(data_path)
            print(f"Total lines in file: {total_lines}")
            print(f"Processing in chunks of {chunk_size} lines")
        
        # Broadcast total_lines to all processes
        total_lines = comm.bcast(total_lines if rank == 0 else None, root=0)
        
        # Process chunks
        for start_line in range(0, total_lines, chunk_size):
            if rank == 0:
                print(f"\nProcessing chunk starting at line {start_line}")
                
            # Read partial graph
            max_node, edges = read_partial_graph(data_path, chunk_size, start_line)
            
            # Create adjacency matrix
            graph = create_graph(max_node, edges)
            
            # Initialize calculator with the partial graph
            calc = ParallelClosenessCentralityCalculator(graph)
            
            # Calculate centrality
            if rank == 0:
                print(f"Calculating centrality for chunk {start_line//chunk_size + 1}...")
            
            calc.calculate_closeness()
            
            # Optional: add a break condition if you want to process only one chunk
            break  # Remove this line if you want to process all chunks
        
        if rank == 0:
            print("Calculation completed successfully!")
            
    except Exception as e:
        if rank == 0:
            print(f"An error occurred: {str(e)}")


"""
Run this code with mpirun and choose number of nodes

mpirun -n 4 python3 GA2/main.py
"""
if __name__ == "__main__":
    test()