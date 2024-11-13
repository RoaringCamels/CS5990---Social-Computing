from utils.helper import ParallelGraphCentralityCalculator
import numpy as np
def main():
    pass

def test():
    graph = np.array([[0, 1, 0, 0, 1], 
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0]])

    calc = ParallelGraphCentralityCalculator(graph)
    closeness_centrality = calc.calculate_closeness_centrality()
    betweenness_centrality = calc.calculate_betweenness_centrality()
    calc.visualize_graph()

    print("Closeness Centrality:", closeness_centrality)
    print("Betweenness Centrality:", betweenness_centrality)

if __name__ == "__main__":
    test()
