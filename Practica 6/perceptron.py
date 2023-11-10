import math ;
import numpy as np;

#This functions returns the output of a perceptron given the weights, the bias and the inputs
def outpt_perceptron(weights, bias,inputs):
    #First we define the lambda function sigmoid
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    #Then we return the output of aplying the lambda at the dot product of the weights and the inputs plus the bias
    return sigmoid(np.dot(weights,inputs) + bias)

def error_fun(X,Y,weights,bias):
    #First we use the previous function to get the output of the perceptron
    output = outpt_perceptron(weights,bias,X)
    #Then we return the error function using the cuadratic error function and normalizing it
    n=len(X)
    return (1/(n))*((Y-output)**2)

#This function will apply gradient descent to train a perceptron
def training_gradient_descent(X,Y,weights,bias,lr):
    #Then we will actualizace the weights 
    weights_act = weights + lr*(error_fun(X,Y,weights,bias)/weights)
    return weights_act




    
    