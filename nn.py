from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Import dataset
data = datasets.load_digits()

# Make array representation of labels
y = []
for x in data.target:
    r = [0,0,0,0,0,0,0,0,0,0]
    r[x] = 1
    y.append(r)
x_train, x_test, y_train, y_test = train_test_split(data.data, y, test_size=0.33, random_state=42)


# Parameters
layer_sizes = [64,10]
epochs = 100
lr = 0.2

# Variables
errors = []


# Activation function, should be the logistic function
def afunc(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the activation function
def afuncDerivative(x):
    return x * (1 - x)


# Initialization of weights
weights = []
for i in range(1, len(layer_sizes)):
    weights.append(2 * np.random.random((layer_sizes[i - 1], layer_sizes[i])) - 1)


# Training function
def train(data, labels, epochs):
    for x in range(epochs):
        # Calculates the values of each neuron in each layer
        layer_output = []
        layer_output.append(data)
        for i in range(len(weights)):
            layer_output.append(afunc(layer_output[-1].dot(weights[i])))

        # Error calculation
        error = 1 / 2 * (layer_output[-1] - y_train) ** 2
        sum_error = np.sum(error)
        errors.append((sum_error))

        # Gradient calculation
        gradients = []
        s1 = (labels - layer_output[-1])
        for x in range(len(weights) - 1, -1, -1):
            grad = s1 * afuncDerivative(layer_output[x + 1])
            gradients.insert(0, grad)
            if x != 0: s1 = gradients[0].dot(weights[x].T)

        # Update weights
        for x in range(len(weights)):
            weights[x] = weights[x]-lr*layer_output[x].T.dot(layer_output[x + 1])





print(weights[0])
train(x_train,y_train,epochs)
print("second", weights[0])
print(errors[0],errors[1],errors[2],errors[3],errors[4])

plt.plot(errors)
plt.show()