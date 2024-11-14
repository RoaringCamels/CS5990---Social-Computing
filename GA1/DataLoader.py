from networkx import Graph

class DataLoader:
    def __init__(self, path, type):
        self.type = type
        self.file_path = path

    def buildNetworkXGraph(self) -> Graph:
        if (self.file_path == None or self.file_path == ' '):
            return None
        network = Graph()
        with open(self.file_path, "r") as file:
                if (self.type == "Amazon"):
                    for index, line in enumerate(file):
                        if line.startswith("#"):
                            continue
                        src, des =  map(int, line.strip().split("\t"))
                        network.add_edge(src,des)
                if (self.type == "Twitch"):
                    for index, line in enumerate(file):
                        if index > 0:
                            src, des =  map(int, line.split(','))
                            network.add_edge(src,des)
        return network