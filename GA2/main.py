from utils.helper import ParallelGraphCentralityCalculator
import numpy as np

def main():
    pass

def test():
    dataPath = f"GA2/data/facebook_combined.txt"
    calc = ParallelGraphCentralityCalculator(dataPath, sample_fraction=0.1)
    calc.calculate_centrality()

if __name__ == "__main__":
    test()
