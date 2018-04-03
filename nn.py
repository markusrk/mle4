#%% loading stuff
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

# seed random
np.random.seed(42)

# Import dataset
data = datasets.load_digits()

# Make array representation of labels
y = []
for x in data.target:
    r = [0,0,0,0,0,0,0,0,0,0]
    r[x] = 1
    y.append(r)
y = np.array(y)

# Normalize features
features = normalize(data.data,axis=0)

# split in traning and testing
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.33, random_state=42)

#%% Important stuff
# Parameters
layer_sizes = [64,10,10]
epochs = 1000
lr = 0.02
accuracy_freq = 1
test_int = 20
dropout_rate = 0.2

# Variables
errors = []
accuracies = []
t_errors = []
t_accuracies = []
sample_points = np.linspace(0,epochs,epochs)
t_sample_points = []


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
def train(data, labels, epochs,t_data,t_labels):
    for y in range(epochs):
        # Calculates the values of each neuron in each layer
        layer_output = []
        layer_output.append(data)
        for i in range(len(weights)):
            layer_output.append(afunc(layer_output[i].dot(weights[i])))

        # Dropout
   #     for i in range(1,len(layer_output)-1):
    #        layer_output[i] *= np.random.binomial(1,1-dropout_rate,layer_output[i].shape)*(1/(1-dropout_rate))

        # Error calculation
        error = 1 / 2 * (layer_output[-1] - y_train) ** 2
        sum_error = np.sum(error)
        errors.append(sum_error/len(labels))

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
        # Sample accuracy
        if y%accuracy_freq == 0:
            correct = 0
            for i in range(len(layer_output[-1])):
                if layer_output[-1][i].argmax() == np.array(labels[i]).argmax():
                    correct += 1
            accuracy = correct/len(layer_output[-1])
            accuracies.append(accuracy)
        # Sample test error and accuracy
        if y%test_int == 0:
            layer_output = []
            layer_output.append(t_data)
            for i in range(len(weights)):
                layer_output.append(afunc(layer_output[i].dot(weights[i])))
            # Accuracy
            correct = 0
            for i in range(len(layer_output[-1])):
                if layer_output[-1][i].argmax() == np.array(t_labels[i]).argmax():
                    correct += 1
            accuracy = correct / len(layer_output[-1])
            t_accuracies.append(accuracy)
            # Error
            error = 1 / 2 * (layer_output[-1] - y_test) ** 2
            sum_error = np.sum(error)
            t_errors.append(sum_error/len(t_labels))
            # Sample_point
            t_sample_points.append((y))

def confusionMatrix(features,labels,title=None):
    # Calculates the values of each neuron in each layer
    layer_output = []
    layer_output.append(features)
    for i in range(len(weights)):
        layer_output.append(afunc(layer_output[i].dot(weights[i])))

    # make matrix
    array = np.zeros((10,10))
    for i in range(len(layer_output[-1])):
        array[layer_output[-1][i].argmax()][labels[i].argmax()] += 1
    # Print plot
    plt.matshow(array)
    if title: plt.title(title)
    plt.show()
    return None

def plots():
    plt.plot(sample_points,errors)
    plt.plot(t_sample_points,t_errors)
    plt.title("Error")
    plt.show()
    plt.plot(sample_points,accuracies)
    plt.plot(t_sample_points,t_accuracies)
    plt.title("Accuracy")
    plt.show()
    return None



train(x_train,y_train,epochs,x_test,y_test)
confusionMatrix(x_train,y_train,"Training data")
confusionMatrix(x_test,y_test,"Test data")
plots()

print("training accuracy:   ",accuracies[-1])
print("test accuracy:       ",t_accuracies[-1])