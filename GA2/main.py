from utils.helper import ParallelGraphCentralityCalculator
import numpy as np
import os

def main():
    pass

def test():
    try:
        dataPath = "GA2/data/facebook_combined.txt"
        
        # Check if the file exists
        if not os.path.exists(dataPath):
            raise FileNotFoundError(f"Data file not found at {dataPath}")
            
        # Initialize the calculator
        calc = ParallelGraphCentralityCalculator(dataPath)
        
        # Sample the graph
        print("Sampling graph...")
        calc.sample_graph(0.05)
        
        # Calculate centrality
        print("Calculating centrality measures...")
        calc.calculate_centrality()
        
        print("Calculation completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test()
