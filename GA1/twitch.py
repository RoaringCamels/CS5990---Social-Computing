from helper import NetworkHelper
from tabulate import tabulate
from DataLoader import DataLoader
import sys
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)

f = open('test_twitch.log', 'w')
backup = sys.stdout
sys.stdout = Tee(sys.stdout, f)

networkHelper = NetworkHelper()
originalNetworkData = DataLoader("large_twitch_edges.csv", "Twitch").buildNetworkXGraph()
GccNetwork = networkHelper.largestConnectedComponent(originalNetworkData)
avgDegreeOrg = None

def original():
    global avgDegreeOrg
    print("========================================= Original Network ======================================")
    avgDegreeOrg = networkHelper.averageDegrees(GccNetwork)

    print("Size: ", networkHelper.getSize(GccNetwork))
    print("Average Degree: ", networkHelper.averageDegrees(GccNetwork))
    print("Average Path Length By Taking 10% Nodes In Original:" , networkHelper.averagePathLengthByPercentage(GccNetwork))
    print("Average Path Length By Taking 20% Nodes In Original:" , networkHelper.averagePathLengthByPercentage(GccNetwork, 0.2))
    print("Average Path Length By Using Random Walk Algorithm:" , networkHelper.averagePathLengthRandomWalk(GccNetwork))
    print("Clustering Coefficient: ", networkHelper.averageClusteringCoefficient(GccNetwork))


def watts():
    wattsModel = networkHelper.watts_strogatz_graph(len(GccNetwork), int(avgDegreeOrg))
    print("========================================= Watts-Strogatz ======================================")
    print("Average Path Length By Taking 10% Nodes In Original:" , networkHelper.averagePathLengthByPercentage(wattsModel))
    print("Average Path Length By Taking 20% Nodes In Original:" , networkHelper.averagePathLengthByPercentage(wattsModel, 0.2))
    print("Average Path Length By Using Random Walk Algorithm:" , networkHelper.averagePathLengthRandomWalk(wattsModel))
    print("Clustering Coefficient: " , networkHelper.averageClusteringCoefficient(wattsModel))

def barabasi():
    barabasiModel = networkHelper.barabasi_albert_graph(len(GccNetwork), (int(avgDegreeOrg) // 2))
    print("========================================= Barabasi-Albert ======================================")
    print("Average Path Length By Taking 10% Nodes In Original:" , networkHelper.averagePathLengthByPercentage(barabasiModel))
    print("Average Path Length By Taking 20% Nodes In Original:" , networkHelper.averagePathLengthByPercentage(barabasiModel, 0.2))
    print("Average Path Length By Using Random Walk Algorithm:" , networkHelper.averagePathLengthRandomWalk(barabasiModel))
    print("Clustering Coefficient: ", networkHelper.averageClusteringCoefficient(barabasiModel))


print("========================================= Twitch ======================================")

original()
barabasi()
watts()


