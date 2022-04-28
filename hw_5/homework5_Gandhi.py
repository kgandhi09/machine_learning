import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import random

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    # Unpack arguments
    start = 0
    end = NUM_HIDDEN*NUM_INPUT
    W1 = w[0:end]
    start = end
    end = end + NUM_HIDDEN
    b1 = w[start:end]
    start = end
    end = end + NUM_OUTPUT*NUM_HIDDEN
    W2 = w[start:end]
    start = end
    end = end + NUM_OUTPUT
    b2 = w[start:end]
    # Convert from vectors into matrices
    W1 = W1.reshape(NUM_HIDDEN, NUM_INPUT)
    W2 = W2.reshape(NUM_OUTPUT, NUM_HIDDEN)
    return W1,b1,W2,b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    return np.hstack((W1.flatten(), b1, W2.flatten(), b2))

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which)).T / 255.
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))

    # TODO: Convert labels vector to one-hot matrix (C x N).
    no_classes = 10
    try:
        vector_labels = np.zeros((labels.size,no_classes))
        vector_labels[np.arange(labels.size),labels] = 1
    except:
        vector_labels = None
    labels = vector_labels.T    
    
    return images, labels

#Given the predictions Y^ and ground truth Y
#Calculates the percent correct accuracy
#Returns the percent correct accuracy value
def fPC (y, yhat):
    return np.count_nonzero(y==yhat)/y.size*100

#Given pre-activation scores z = W.X + b
#calculates relu function value y = max{0,z} 
#returns relu function value
def relu(z):
    z[z<=0]= 0
    return z

#Given pre-activation scores z = W.X + b
#calculates relu function value y = max{0,z} 
#returns relu function value
def relu_prime(z):
    z[z<=0]= 0
    z[z>0] = 1
    return z

#Given pre-activation scores z = W.X + b
#Calculates a probability distribution for z
#Returns the probability distribution for z
def softmax_activation(z):
    exp_z = np.exp(z)
    sum_z = np.sum(exp_z, axis=1).reshape(len(z),1)
    y_hat = exp_z/sum_z
    return y_hat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss, accuracy,
# as well as the intermediate values of the NN.
def fCE (X, Y, w):
    Y = Y.T
    W1, b1, W2, b2 = unpack(w)

    b1 = np.reshape(b1, (NUM_HIDDEN,1))
    b2 = np.reshape(b2, (NUM_OUTPUT,1))

    z_1 = W1.dot(X) + b1
    h_1 = relu(z_1)
    z_2 = W2.dot(h_1) + b2
    y_hat = softmax_activation(z_2)

    inner_math = Y.T*np.log(y_hat)
    inner_math = np.sum(inner_math,axis=1)
    cost = (np.mean(inner_math)*-1)
    
    return cost, z_1, h_1, W1, W2, y_hat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    
    W1, b1, W2, b2 = unpack(w)

    #forward propogation equations
    b1 = np.reshape(b1, (NUM_HIDDEN,1))
    b2 = np.reshape(b2, (NUM_OUTPUT,1))

    z_1 = W1.dot(X) + b1
    h_1 = relu(z_1)
    z_2 = W2.dot(h_1) + b2
    y_hat = softmax_activation(z_2)
    print(y_hat)
    Y = Y.T

    #background propogation equations
    grad_b2 = np.mean((y_hat - Y.T), axis=1) #(10,)
    grad_w2 = (y_hat - Y.T).dot(h_1.T) # (10, 40)

    g_T = grad_b2.T.dot(W2) * (relu_prime(z_1).T) #(n,40)
    g = g_T.T

    grad_b1 = np.mean(g, axis=1) #(40,)
    grad_w1 = g.dot(X.T) #(40,784)
    
    grad = pack(grad_w1, grad_b1, grad_w2, grad_b2)
    
    return grad

#Given input images Xtilde, and batch_size
#Divides the training data into random batches based on batch_size
#Returns the list of randomized batches of training data [(M+1)xbacth_size, ...]
def prepare_training_data(Xtilde, Y, batch_size):
    no_batches = Xtilde.shape[1]/batch_size
    image_batches = np.hsplit(Xtilde, no_batches)
    label_batches = np.hsplit(Y, no_batches)
    return [image_batches, label_batches]

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w):
    no_epochs = 10
    batch_size = 60000
    learning_rate = 0.1
    no_of_batches = (int)(trainX.shape[1]/batch_size)
    
    training_data = prepare_training_data(trainX, trainY, batch_size)
    training_images = training_data[0]
    training_labels = training_data[1]

    random_batch_list = []
    
    counter = 0
    while(counter < no_of_batches):
        n = random.randint(1,no_of_batches)-1
        if(n not in random_batch_list):
            random_batch_list.append(n)
            counter += 1

    #unpacking different terms from weights
    w1, b1, w2, b2 = unpack(w)
    
    for epoch in range(no_epochs):
        mini_batch_counter = 1
        for random_batch_no in random_batch_list:
            training_images_batch = training_images[random_batch_no]
            training_labels_batch = training_labels[random_batch_no]
            
            grad_vector = gradCE(training_images_batch, training_labels_batch, w)
            grad_w1, grad_b1, grad_w2, grad_b2 = unpack(grad_vector)

            b1 = b1.reshape(b1.shape[0], 1)
            b2 = b2.reshape(b2.shape[0], 1)
            grad_b1 = grad_b1.reshape(grad_b1.shape[0], 1)
            grad_b2 = grad_b2.reshape(grad_b2.shape[0], 1)

            w1 = w1 - learning_rate*grad_w1 
            b1 = b1 - learning_rate*grad_b1
            w2 = w2 - learning_rate*grad_w2
            b2 = b2 - learning_rate*grad_b2

            b1 = b1.reshape(b1.shape[0],)
            b2 = b2.reshape(b2.shape[0],)
            w = pack(w1,b1,w2,b2)

            cost,z_1,h_1, w1, w2, y_hat = fCE(training_images_batch, training_labels_batch, w)
            print("Epoch: " + str(epoch+1) +  " Batch no: " + str(mini_batch_counter) + " Cross-Entropy Loss: " + str(cost))
            mini_batch_counter += 1


if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    # idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    # print("Numerical gradient:")
    # print(scipy.optimize.approx_fprime(w, lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_)[0], 1e-10))
    # print("Analytical gradient:")
    # print(gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w))
    # print("Discrepancy:")
    # print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_)[0], \
    #                                 lambda w_: gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[:,idxs]), w_), \
    #                                 w))

    # # Train the network using SGD.
    train(trainX, trainY, testX, testY, w) 
