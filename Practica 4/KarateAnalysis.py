#First que need to make all the imports
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#We are going to create a class that will allow us to appl algorithms and load the graph from the .txt file
class AlgorithmsAndCommunities:
    
    #The innit will take the path of the file and load it as a graph
    def __init__(self, path):
        graphArray = np.loadtxt(path)
        graph = nx.from_numpy_array(graphArray)
        self.graph = graph
    
    # calculate the betweenness centrality of the graph
    def betweenness_centrality(self):
        #We use the prebuilt functions on networkx
        betweenness=nx.betweenness_centrality(self.graph)
        print("\n Betweenness centrality: ", betweenness)
        return betweenness
    

#We will create an object of our class to test the functions
graph=AlgorithmsAndCommunities("Practica 4\graph_data.txt")
graph.betweeness_centrality()
