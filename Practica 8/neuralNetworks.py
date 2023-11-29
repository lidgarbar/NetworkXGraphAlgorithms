import numpy as np
import torch 
import torch.nn as nn
import math

#Exercise 1 (Graph Generation)
def sbm(n, c, p_intra, p_inter):
    """
    n: number of nodes
    c: number of communities
    p_intra: intra-community probability
    p_inter: inter-community probability
    """
    
    #Assign a community to each node.
    #The community must be a list with the label of each community for each node.
    #Communities should be balanced.
    community = np.arange(c).repeat(n // c) #Repeat each community n // c times.
    
    #Ensure the vector has length n.
    community = community[0:n]
    
    #Make it a column vector.
    community = np.expand_dims(community, 1)
    
    #Generate a matrix of booleans indicating if two nodes are in the same community.
    intra = community == community.T
    
    #Generate a matrix of booleans indicating if two nodes are in different communities.
    inter = np.logical_not(intra) #Invert the boolean values.
    
    #Generate a matrix with random entries between 0 and 1.
    random = np.random.rand(n, n) 
    
    #Initialize the adjacency matrix of the graph with a matrix of zeros.
    graph = np.zeros((n, n))
    
    #Assign an edge when the probability condition is met using the random matrix for intra-community.
    #Being symmetric,we will update only the upper triangular part.
    graph[np.triu(intra)] = (random[np.triu(intra)] < p_intra).astype(int) #If the condition is met, assign a 1, otherwise a 0
    
    # Assign an edge when the probability condition is met using the random matrix for inter-community.
    # Being symmetric, we will update only the upper triangular part.
    graph[np.triu(inter)] = (random[np.triu(inter)] < p_inter).astype(int) #If the condition is met, assign a 1, otherwise a 0
    
    #Make the matrix symmetric
    graph += graph.T
    
    return graph

#Example:
S = sbm(n=50, c=5, p_intra=0.6, p_inter=0.2)
#print("Exercise 1: ",S)

#Exercise 2 (Normalize by the Largest Eigenvalue)
def normalize(graph):
    """
    Normalize the graph Laplacian by dividing by the maximum eigenvalue.
    """
    
    #Calculate the eigenvalues
    eigenvalues, _ = np.linalg.eig(graph) #The second argument is not needed
    
    #Normalize by dividing by the eigenvalue with the largest absolute value.
    normalized_graph = graph / np.max(np.abs(eigenvalues))
    
    return normalized_graph

#Example:
S = normalize(S)
#print("Excercise 2: ",S)

#Exercise 3 (Data Generation for Diffusion Process)
import numpy as np

def generate_diffusion(Graph, n_samples, n_sources):
    # Calculate the number of nodes in the graph
    n = Graph.shape[0]

    # Initialize a tensor of zeros to store the samples
    # of size n_samples x n x T+1 x 1.
    z = np.zeros((n_samples, n, 5, 1))

    for i in range(n_samples):
        # Take n_sources randomly from the n nodes
        sources = np.random.choice(n, n_sources, replace=False)

        # Define z_0 for each sample
        z[i, sources, 0, 0] = np.random.uniform(0, 1, n_sources)

    # Mean and variance of the noise
    mu = np.zeros(n)
    sigma = np.eye(n) * 1e-3

    for t in range(4):
        # Generate the noise
        noise = np.random.multivariate_normal(mu, sigma, n_samples)

        # Generate z_t
        z[:, :, t + 1] = np.matmul(Graph, z[:, :, t]) + np.expand_dims(noise, -1)

    # Transpose the dimension so that it is
    # n_samples x time x n x 1
    z = z.transpose((0, 2, 1, 3))

    # "Squeeze" the dimension as it only has dimension 1.
    return z.squeeze()


#Example usage:
graph = sbm(n=50, c=5, p_intra=0.6, p_inter=0.2)
diffusion_matrix = generate_diffusion(graph,2100,10)
#print("Excercise 3: ",diffusion_matrix)


#Exercise 4 (Data Acquisition)

def data_from_diffusion(z):
    # Permute the samples of z
    z = np.random.permutation(z)

    # Define the output tensor
    y = np.expand_dims(z[:, 0, :], 1)

    # Initialize the input tensor as a matrix
    # of zeros with the same dimension as y.
    x = np.zeros_like(y)

    # Update the input tensor as x = z_4
    for i, sample in enumerate(z):
        x[i] = sample[4]

    # Squeeze the time dimension.
    return x.squeeze(), y.squeeze()

#Split into training and test sets
#zTrain and zTest are available using the code from Exercise 3
#Example usage:
graph = sbm(n=50, c=5, p_intra=0.6, p_inter=0.2)
zTrain = generate_diffusion(graph,2100,10)
zTest = generate_diffusion(graph,2100,10)
xTrain, yTrain = data_from_diffusion(zTrain)
xTest, yTest = data_from_diffusion(zTest)

#print("Excercise 4: ",xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)

#Convert to PyTorch tensors
xTrain = torch.tensor(xTrain)
yTrain = torch.tensor(yTrain)
xTest = torch.tensor(xTest)
yTest = torch.tensor(yTest)

#Exercise 5 (Graph Filters)
def FilterFunction(h, S, x):
    K = h.shape[0]  # Filter order
    B = x.shape[0]  # Batch size
    N = x.shape[1]  # Number of nodes
    x = x.reshape([B, 1, N])
    S = S.reshape([1, N, N])
    z = x
    
    for k in range(1, K):
        # Diffusion step, S^k*x
        x = torch.matmul(x, S)
        xS = x.reshape([B, 1, N])
        
        # Concatenate S^k*x to the tensor z
        z = torch.cat(( z,xS), dim=1)
    
    # Multiply z and h
    y = torch.matmul(z.permute(0, 2, 1).reshape([B, N, K]).double(), h.double())

    return y

#The graph filter is implemented as a PyTorch module
class GraphFilter(nn.Module):
    def __init__(self, graph, k):
        # Initialize
        super(GraphFilter, self).__init__()
        
        # Save hyperparameters
        self.graph = torch.tensor(graph)
        self.n = graph.shape[0]
        self.k = k
        
        # Define and initialize weights
        self.weight = nn.Parameter(torch.randn(self.k))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.k)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return FilterFunction(self.weight, self.graph, x)
    
# Example:
gF = GraphFilter(S, 8)
#print("Exercise 5: ",gF.forward(xTrain))