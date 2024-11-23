from utils.helper import ParallelClosenessCentralityCalculator
from utils.utils import read_graph
from mpi4py import MPI

def test():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Read graph (only root process needs to read)
    if rank == 0:
        print(f"Process {rank}: Reading graph...")
        
    # Set the path to your data file
    data_path = "GA2/data/fb.txt"
    
    # Read the graph
    graph = read_graph(data_path, 1.0)
    
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