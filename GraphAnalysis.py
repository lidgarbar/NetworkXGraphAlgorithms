#First que need to make all the imports
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#We are going to load the graph that we have stored as an aray in the .txt file
def load_graph(path):
    graphArray = np.loadtxt(path)
    return nx.from_numpy_array(graphArray)

graph= load_graph('graph_data.txt')

#After loading the graph we will look at the degree of the  and plot the degree for each node
def represent_degrees(g):
    degrees=graph.degree
    plt.hist(degrees)
    plt.show()
    return print("\n Degrees: ", degrees)

#Calculate the clustering coefficient and the mean clustering coefficient
def clustering_coefficients(g):
    #We use the prebuilt functions on networkx
    clustering=nx.clustering(g)
    average=nx.average_clustering(g)
    
    #After that we will print out the answer
    print("\n Clustering coefficient:         ", clustering)
    return print("\n Average Clustering coefficient: ", average)
    
clustering_coefficients(graph)

#We are going to determine if there is a clique bigger than 3 in a directed graph 
#mirar algún agujero estructural (mirar si coeficiente de clustering distinto de 1 )
def clique_bigger_than_3(g):
    cliques=nx.find_cliques(g)
    for clique in cliques:
        if len(clique)>3:
            print("\n There is a clique bigger than 3: ", clique)
            return clustering_coefficients(nx.subgraph(graph,clique))        
    return print("\n There is no clique bigger than 3")

clique_bigger_than_3(graph)
#As we can see it is obvius that if we apply the clustering coefficient function in a clique the 
# result will be of 1 on each node an averge

#We will seek now for the biggest k-core and represent the k-shell
def k_core(g):
    k_core=nx.k_core(g)
    k_shell=nx.core_number(g)
    #nx.draw(k_shell , with_labels = True , font_weight = 'bold')
    #plt.savefig( "graph_k_shell.png" )
    #plt.show()
    print("\n The biggest k-core is: ", k_core)
    return print("\n The k-shell is: ", k_shell)

k_core(graph)

#We will count the k_components and specifically 2_components and 3-components of the graph
#Cuando nos sale dos significa que hay dos subgrupos que solo hay un camino entre ellos (los dos clusters)
def k_components(g):
    k_components=nx.k_components(g)
    print("\n The k-components are: ", k_components)
    return print("\n The 2-components are: ", k_components[2], "\n The 3-components are: ", k_components[3])


k_components(graph)

#Juntamos todo en el contexto de que hay john y mr.hi y mirar si hay akgu clique que contenga a los dos, o quizas 
#segun lo que utilicemos para ver los cluster nos puede aparecer mas o menos, en este caso lo tenemos claro, son dos
#Tengo que revisar porque no me deja hacerle un plot y como sacar el pyplot ppr pantalla
