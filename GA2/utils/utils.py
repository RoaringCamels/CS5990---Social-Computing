import networkx as nx

def read_graph(file_path, percentage=1.0):
    """
    Read graph from file and return NetworkX graph object
    
    Time Complexity:
    ---------------
    O(E) where E is the total number of edges in the file
    - First pass: O(E) to count total lines
    - Second pass: O(p*E) where p is the percentage (0 to 1)
    - Graph construction: O(p*E) for adding edges
    Total: O(E) as p ≤ 1
    
    Space Complexity:
    ----------------
    O(V + p*E) where V is number of vertices and E is number of edges
    - O(V) for storing vertices in the graph
    - O(p*E) for storing edges based on percentage read
    - O(1) for other variables (total_lines, current_line, etc.)
    
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
    
    Notes:
    ------
    - The method performs two passes over the file:
      1. First pass counts total valid lines (O(E))
      2. Second pass reads selected percentage of edges (O(p*E))
    - Memory usage scales with number of unique vertices and selected edges
    - Graph construction is incremental, adding edges one by one
    """
    # Input validation - O(1)
    if not 0 < percentage <= 1:
        raise ValueError("Percentage must be between 0 and 1")
    
    # First pass: Count total valid lines
    # Time: O(E) to read all lines
    # Space: O(1) for counter
    total_lines = 0
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):  # Skip comments
                total_lines += 1
    
    # Calculate lines to read - O(1)
    lines_to_read = int(total_lines * percentage)
    
    # Second pass: Read and construct graph
    # Time: O(p*E) for reading and adding edges
    # Space: O(V + p*E) for graph construction
    G = nx.Graph()  # Initialize empty graph
    current_line = 0
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):  # Skip comments
                if current_line < lines_to_read:
                    # O(1) operations per line:
                    # - split string
                    # - convert to int
                    # - add edge to graph
                    source, target = map(int, line.strip().split())
                    G.add_edge(source, target)
                    current_line += 1
                else:
                    break
    
    # Print statistics - O(1) operations
    print(f"Read {current_line} edges out of {total_lines} total edges ({percentage*100:.1f}%)")
    print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    return G

"""
Performance Analysis:
-------------------
1. File Reading:
   - Two passes through file required
   - First pass: O(E) to count lines
   - Second pass: O(p*E) to read selected edges
   
2. Memory Usage:
   - Graph storage: O(V + p*E)
   - Temporary variables: O(1)
   - No additional data structures needed
   
3. Scalability:
   - Linear with file size for time complexity
   - Memory scales with unique vertices and selected edges
   - Percentage parameter allows memory-constrained processing
   
4. Bottlenecks:
   - File I/O operations (two passes required)
   - Graph construction (adding edges one by one)
   
5. Optimization Notes:
   - Could be optimized for very large files by:
     * Using single pass with reservoir sampling
     * Batch processing of edges
     * Memory-mapped file reading
"""