import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Setting the seed to reproduce the results. You can remove this line
np.random.seed(1)

# Load the data set from sklearn
def load_data():
    digits = load_digits()
    X = digits.data
    m = X.shape[0]
    Y = digits.target.reshape(m, 1) # Need to reshape as numpy will return a 1D array otherwise
    return X, Y

# Use this if you want to scale the image
def scale(x, y, factor = 255):
    return x/factor, y/factor

# Randomly shuffle the set and split it into test and train set
def split(X, Y, ratio):
    x, y = shuffle(X, Y)
    return train_test_split(x, y, test_size = ratio)

# Visualize n images from the dataset
def visualize(X, Y, n):
    for i in range(n):
        plt.imshow(X[i, :].reshape(8, 8))
        print(Y[i])
        plt.show()

# Perform one hot encoding on the labels
def encode(z):
    onehot = OneHotEncoder()
    return onehot.fit_transform(z).toarray()

# Softmax activation to calculate the probabilities
def softmax(z):
    exps = np.exp(z)
    return exps/exps.sum(axis = 1, keepdims = True)

# Calculate the cross-entropy loss
def cost(y, a):
    return -np.mean(y*np.log(a) + (1-y)*np.log(1-a))

# Randomly initialize the parameters
def init_parameters(output_size, input_size):
    w = np.random.randn(output_size, input_size) * 0.01
    b = np.zeros((output_size, 1))
    return w, b

# Move forward on the training process. Calculates the activation for a given input
def forward(x, w, b):
    z = x.dot(w.T) + b.T
    a = softmax(z)
    return a

# Calculate the gradients for the weights and biases
def backward(x, y, a):
    m = y.shape[0]
    dz = a - y
    dw = 1/m * np.dot(dz.T, x)
    db = np.mean(dz, axis = 0, keepdims = True)
    return dw, db


# Train the classifier
# Set verbose = True for training progess
# Set return_costs = True to return the costs at each iteration
def train(x_train, y_train, learning_rate, iterations, return_costs = False, verbose = False):
    w, b = init_parameters(10, 64)
    m = x_train.shape[0]
    costs = []

    if verbose:
        print("Starting....")

    for i in range(iterations):
        if verbose == True:
            if((i+1) % int(iterations/5) == 0):
                print("Iteration: ", i + 1)
                print(cost(y_train, a))

        a = forward(x_train, w, b)
        dw, db = backward(x_train, y_train, a)

        w = w - learning_rate*dw
        b = b - learning_rate*db.T

        costs.append(cost(y_train, a))

    # Save the weights and biases so that they can be loaded later
    np.save('weight', w)
    np.save('bias', b)

    if return_costs == True:
        return w, b, costs
    else:
        return w, b

# Predict output for a given input. You can tweak the threshold to get the required confidence
THRESHOLD = 0.8
def predict(x, w, b):
    a = forward(x, w, b) >= THRESHOLD
    return np.argmax(a, axis = 1).reshape(x.shape[0], 1)

# Calculates the accuracy of the predictions
def accuracy(y, y_pred):
    return np.mean(y_pred == y)

# Main driver function to test the program
def main():
    X, Y = load_data()

    x_train, x_test, y_train, y_test = split(X, Y, 0.2)
    y_train_original = y_train.copy()

    print('Train Set Size: ', y_train.shape[0])
    print('Test Set Size: ', y_test.shape[0])

    '''Uncomment these lines to train the model'''
    # y_train = encode(y_train)
    # w, b, costs = train(x_train, y_train, learning_rate = 0.09, iterations = 2000, return_costs = True, verbose = True)
    # plt.plot(costs)
    # plt.show()

    print("\nTraining is done!\n")

    w = np.load('weight.npy')
    b = np.load('bias.npy')

    y_train_pred = predict(x_train, w, b)
    y_pred = predict(x_test, w, b)
    print("Train Accuracy: ", accuracy(y_train_original, y_train_pred))
    print("Test Accuracy: ", accuracy(y_test, y_pred))

if __name__ == '__main__':
	main()
