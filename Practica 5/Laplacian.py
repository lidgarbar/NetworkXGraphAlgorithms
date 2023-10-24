import numpy as np;
import matplotlib.pyplot as plt;
import networkx as nx;

import spicy as sp;

class Laplacian_analysis:

    #Given a cloud of points, we will calculate the k-nearest neighbours of each point and build aristas between them in a graph
    def k_nearest(self,points, k):
        #We must have a list of neighbours for each point
        neighbours=[]
        
        #We will calculate the distance to all the other points
        distances=sp.spatial.distance_matrix(points, points)
        
        #For each pint we will take the k-nearest neighbours
        for point in range(len(points)):
            distance=distances[point]
            #We will sort the distances
            distance.sort()
            #We will take the first k distances
            k_distances=distance[:k]
            #We will store the k-nearest neighbours of each point
            neighbours.append(k_distances)
            
        #After that we will build the graph
        graph=nx.Graph()
        #We will add the nodes
        graph.add_nodes_from(points)
        
        #We will add the edges, for each point 
        for i in range(len(points)):
            #We will take the k neighbours of each point
            k=len(neighbours[i])
            #We will add an edge between the point and each neighbour
            for j in range(k):
                graph.add_edge(points[i], neighbours[i][j])
                
        #We will return the graph
        return graph

    #Given a list of edges we will build the Laplacian matrix
    def laplacian_matrix(self,edges):
        edges=list(edges)
        #First we need to create a matrix to store the values
        matrix=np.zeros((len(edges), len(edges)))
        
        #For each edge, evaluate the degree and assign values to the matrix
        for i, edge in enumerate(edges):
            node1, node2 = edge

            #First we check if node1 and node2 are equal using np.all()
            if np.any(node1 == node2):
                #Calculate the degree counting the number of neighbours of the node
                degree = len(list(nx.neighbors(graph, node1)))
                #Assign the degree to the diagonal element
                matrix[i, i] = degree 
                
            else:
                # Check if the edge (node1, node2) is present in the list and assign -1 to the element
                if edge in edges:
                    # Find the index of the edge in the list of edges
                    index = edges.index(edge)
                    # Assign -1 to the element
                    matrix[i, index] = -1
                    
                    
                #Otherwise, assign 0 to the element 
                else:
                    matrix[i, i] = 0


        return matrix

    #Given a Laplacian matrix and a number of clusters (k) we will calculate the eigenvalues and eigenvectors
    def matrix_eigenvectors(self,matrix, clusters):
        #We will calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors=np.linalg.eig(matrix)
        
        #We will sort the  eigenvectors
        eigenvectors.sort()
        #We will take the first k eigenvectors
        eigenvectors=eigenvectors[:clusters]
        

        
        #We will store the eigenvectors in the  columns of a matrix
        for i in range(len(eigenvectors)):
            matrix[:,i]=eigenvectors[i]
            
        return matrix

    #Using the other functions we will build the spectral embending given a cloud of points and a number of clusters (k-neighbours)
    def spectral_embending(self,points, clusters):
        #We will calculate the k-nearest neighbours
        graph=self.k_nearest(points, clusters)
        
        #We will build the Laplacian matrix
        matrix=self.laplacian_matrix(graph.edges)
        
        #We will calculate the matrix and the eigenvalues and eigenvectors
        matrix=self.matrix_eigenvectors(matrix, clusters)

        #Now we need to normalize the rows of the matrix (L2 normalization)
        matrix = matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]   
        
        return matrix

#Now we will test the functions
laplacian=Laplacian_analysis()
#We will use the iris.txt file to test the functions
iris=np.loadtxt("Practica 5\iris.txt")
# Convert the data points to tuples (hashable) to avoid errors
iris = [tuple(point) for point in iris]

#Show the results of the k-nearest neighbours
graph=laplacian.k_nearest(iris, 3)

#Show the results of the Laplacian matrix
matrix=laplacian.laplacian_matrix(graph.edges)
print(matrix)

#We will apply the spectral embending
embending=laplacian.spectral_embending(iris, 3)
#We need to represent all of the embending
plt.scatter(embending[:,0], embending[:,1], c=embending[:,2])
plt.show()

