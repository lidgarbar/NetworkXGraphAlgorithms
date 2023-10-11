#First que need to make all the imports
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#We are going to create a class that will allow us to analyse the graph loaded from the .txt file
class GraphAnalysis:
    #We are going to load the graph that we have stored as an aray in the .txt file
    def __init__(self, path):
        graphArray = np.loadtxt(path)
        graph = nx.from_numpy_array(graphArray)
        self.graph = graph

    #After loading the graph we will look at the degree of it and plot the degree for each node
    def represent_degrees(self):
        degrees=self.graph.degree
        plt.hist(degrees)
        plt.show()
        return print("\n Degrees: ", degrees)

    #Calculate the clustering coefficient and the mean clustering coefficient
    def clustering_coefficients(self):
        #We use the prebuilt functions on networkx
        clustering=nx.clustering(self.graph)
        average=nx.average_clustering(self.graph)
        
        #After that we will print out the answer
        print("\n Clustering coefficient:         ", clustering)
        return print("\n Average Clustering coefficient: ", average)
        
    #We are going to determine if there is a clique bigger than 3 in a directed graph 
    #We also need to see if there is a structural hple in the clique (clustering_coefficient not 1 )
    def clique_bigger_than_3(self):
        cliques=nx.find_cliques(self.graph)
        for clique in cliques:
            if len(clique)>3:
                print("\n There is a clique bigger than 3: ", clique)
            #For each node in the clique we will check if there is a structural hole (clustering coefficient not 1)
            for node in clique:
                if nx.clustering(self.graph)[node]!=1:
                    print("\n There is a structural hole in node: ", node, "(",nx.clustering(self.graph)[node],") in the clique: ", clique)
                return self.clustering_coefficients()
        return print("\n There is no clique bigger than 3")

    #We will seek now for the biggest k-core and represent the k-shell
    def k_core(self):
        k_core=nx.k_core(self.graph)
        k_shell=nx.core_number(self.graph)
        #We will now draw the k-shell
        nx.draw(self.graph, nodelist=k_shell, node_color='r', with_labels=True)
        plt.show()
        print("\n The biggest k-core is: ", k_core)
        return print("\n The k-shell is: ", k_shell)

    #We will count the k_components and specifically 2_components and 3-components of the graph
    #If we have two components there is only one path between them so we will print this path
    def k_components(self):
        k_components=nx.k_components(self.graph)
        print("\n The k-components are: ", k_components)
        
        #If there is a 2 components we will print the path between the two sets of nodes
        if len(k_components[2])==2:
            k1=k_components[2][0];k2=k_components[2][1]
            
            #We will see the connecting nodes between the two sets
            connecting_nodes = set(self.graph.nodes).intersection(k1, k2)
            #If there is a connecting node that implies that the distance is 1
            if(len(connecting_nodes)>0):
                print(f"Distance is 1 using theese connecting nodes: {connecting_nodes}")
                
            #If not we will use the shortest path function to find the path and the distance 
            else:
               print(f"Shortest path: {nx.shortest_path(self.graph, k1[0], k2[0])}") 
               print(f"Distance is {nx.shortest_path_length(self.graph, k1[0], k2[0])}")

        return print("\n The 2-components are: ", k_components[2], "\n The 3-components are: ", k_components[3])

#Lets test all the functions first we need to create an object of the class
analysis= GraphAnalysis("Practica 3\graph_data.txt")

#We will call all the functions using the object of the class
def test_class(an):
    an.clustering_coefficients()
    an.clique_bigger_than_3()
    #As we can see it is obvius that if we apply the clustering coefficient function in a clique the 
    # result will be of 1 on each node an averge
    an.k_core()
    an.k_components()

#We will use the clustering coefficient to separate the graph into two clusters and draw it
def john_mrHi(graph):
    #We will create a new graph with the same nodes as the original one
    john_mrHi=nx.Graph()
    john_mrHi.add_nodes_from(graph)
    #We will create a list with the clustering coefficient of each node
    clustering=nx.clustering(graph)
    #We will create a list with the nodes that have a clustering coefficient of 1
    john=[]
    mrHi=[]
    for node in clustering:
        if clustering[node]==1:
            john.append(node)
        else:
            mrHi.append(node)
    #We will add the edges to the new graph
    john_mrHi.add_edges_from(graph.edges(john))
    john_mrHi.add_edges_from(graph.edges(mrHi))
    
    #We will draw the graph
    nx.draw(john_mrHi, nodelist=john, node_color='r', with_labels=True)
    nx.draw(john_mrHi, nodelist=mrHi, node_color='b', with_labels=True)
    plt.show()
    
    #We will return the new graph
    return john_mrHi    


#Deploying the test of the class and also the john and mrHi function
john_mrHi= john_mrHi(analysis.graph)
test_class(analysis)