from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Import dataset
data = datasets.load_digits()

# Make array representation of labels
y = []
for x in data.target:
    r = [0,0,0,0,0,0,0,0,0,0]
    r[x] = 1
    y.append(r)

# Normalize features
features = normalize(data.data,axis=0)

# split in traning and testing
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.33, random_state=42)


# Parameters
layer_sizes = [64,10]
epochs = 1000
lr = 0.02
accuracy_freq = 1

# Variables
errors = []
accuracies = []


# Activation function, should be the logistic function
def afunc(x):
    return 1/(1 + np.exp(-x))


# Derivative of the activation function
def afuncDerivative(x):
    return x*(1-x)


# Initialization of weights
weights = []
for i in range(1, len(layer_sizes)):
    weights.append(np.random.normal(0,1,(layer_sizes[i - 1], layer_sizes[i])))


# Training function
def train(data, labels, epochs):
    for y in range(epochs):
        # Calculates the values of each neuron in each layer
        layer_output = []
        layer_output.append(data)
        for i in range(len(weights)):
            layer_output.append(afunc(layer_output[i].dot(weights[i])))

        # Error calculation
        error = 1 / 2 * (layer_output[-1] - y_train) ** 2
        sum_error = np.sum(error)
        errors.append(sum_error)

        # Gradient calculation
        gradients = []
        s1 = (labels - layer_output[-1])
        for i in range(len(weights) - 1, -1, -1):
            grad = s1 * afuncDerivative(layer_output[i + 1])
            gradients.insert(0, grad)
            if i != 0: s1 = gradients[0].dot(weights[i].T)

        # Update weights
        for i in range(len(weights)):
            weights[i] = weights[i]+lr*layer_output[i].T.dot(gradients[i])

        if y%accuracy_freq == 0:
            correct = 0
            for i in range(len(layer_output[-1])):
                if layer_output[-1][i].argmax() == np.array(labels[i]).argmax():
                    correct += 1
            accuracy = correct/len(layer_output[-1])
            accuracies.append(accuracy)



train(x_train,y_train,epochs)
#for x in range(50):
#    print(errors[x])

print()
print()
for x in range(300):
    print(accuracies[x])

plt.plot(errors)
plt.show()
plt.plot(accuracies)
plt.show()