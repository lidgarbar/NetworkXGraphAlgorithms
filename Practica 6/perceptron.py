#First we make all the imports above the code
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_circles  # Add this line
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Exercise 1: We define the sigmoid function and the output of the perceptron
def sigmoid(x):
    # Sigmoid activation function formula
    return 1 / (1 + np.exp(-x))

def output_perceptron(W, b, x):
    #Perceptron output calculation with sigmoid activation
    z = np.dot(x, W) + b
    return sigmoid(z)

#Exercise 2: We define the mean squared error function to calculate the error of the perceptron
def mean_squared_error(Y_true, Y_pred):
    # Mean Squared Error calculation formula
    return np.mean((Y_true - Y_pred)**2)

#Exercise 3: In order to train the model we are using the gradient descent algorithm
def gradient_descent_update(X, Y, W, b, learning_rate):
    #Number of samples
    m = len(X)
    #Compute predictions using the current weights and bias and the output_perceptron function
    predictions = output_perceptron(W, b, X)
    #Compute the error between predictions and actual values using the mean_squared_error function
    error = predictions - Y

    #Compute the gradient with respect to weights (dW) and bias (db) using the formulas of the derivatives
    dW = np.dot(X.T, error * predictions * (1 - predictions)) / m
    db = np.sum(error * predictions * (1 - predictions)) / m

    #Update weights and bias using the gradient and learning rate 
    W -= learning_rate * dW
    b -= learning_rate * db
    
    #Return the updated weights and bias of the trained perceptron
    return W, b

#Exercise 4: We define the Perceptron class with the train and predict methods
class Perceptron:
    #The Constructor must have the input_size parameter to initialize the weights and bias of the perceptron
    def __init__(self, input_size):
        #Initialize random weights and bias
        self.W = np.random.rand(input_size)
        self.b = np.random.rand(1)

    #The train method must have the X_train, Y_train, epochs and learning_rate parameters 
    def train(self, X_train, Y_train, epochs, learning_rate):
        #For each epoch we will update the weights and bias using the gradient_descent_update function
        for epoch in range(epochs):
            self.W, self.b = gradient_descent_update(X_train, Y_train, self.W, self.b, learning_rate)

    #The predict method must have the X parameter 
    def predict(self, X):
        #Make predictions using the trained perceptron and the output_perceptron function
        return output_perceptron(self.W, self.b, X)

#Generate datasets using the make_classification and make_circles functions as it says in the exercise
X1, Y1 = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
X2, Y2 = make_circles(n_samples=200, noise=0.05, random_state=42)

#Split datasets into training and test sets using the train_test_split function
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=42)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.3, random_state=42)

#Create and train the perceptron for the first dataset
perceptron1 = Perceptron(input_size=2)
perceptron1.train(X1_train, Y1_train, epochs=1000, learning_rate=0.01)

#Create and train the perceptron for the second dataset
perceptron2 = Perceptron(input_size=2)
perceptron2.train(X2_train, Y2_train, epochs=1000, learning_rate=0.01)

#Evaluate performance on test sets
Y1_pred = perceptron1.predict(X1_test)
Y2_pred = perceptron2.predict(X2_test)

#Calculate the error on the test sets using the mean_squared_error function
error1 = mean_squared_error(Y1_test, Y1_pred)
error2 = mean_squared_error(Y2_test, Y2_pred)

#Print the error on the test sets
print("Error on dataset 1:", error1)
print("Error on dataset 2:", error2)

#Exercise 5: We will use tensorflow to build a neural network with 2 hidden layers and 1 output layer
#Generate a synthetic dataset
X, label = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)

#Split the dataset into training and test sets
X_train, X_test, label_train, label_test = train_test_split(X, label, test_size=0.3, random_state=42)

#Build the model with the input layer, 2 hidden layers and the output layer
input_dim = X.shape[1]
bias = True
activation_function = 'sigmoid'

inputs = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(3, use_bias=bias, activation=activation_function)(inputs)
x = tf.keras.layers.Dense(2, use_bias=bias, activation=activation_function)(x)
predictions = tf.keras.layers.Dense(input_dim, use_bias=bias, activation='softmax', name='final_output')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

#Display model summary
model.summary()

#Compile the model
lr = 0.03
loss_function = 'mean_squared_error'
model.compile(optimizer="adamax", loss=loss_function, metrics=['accuracy'])

#Train the model
model.fit(X_train, label_train, epochs=500, batch_size=5, verbose=True)

#Evaluate the model on the test set
model.evaluate(X_test, label_test)
