import numpy as np
import matplotlib.pyplot as plt

def reshapeAndAugment(input_images):
    ones = np.ones((1, input_images.shape[0]))
    input_images = input_images.T
    input_images = np.append(input_images, ones, axis=0)
    return input_images

#Given input data X and weights W
#Calculate the preactivation scores, i.e., Z = X^t * W
#Returns Z
def preActivationScores(Xtilde, Wtilde):
    z = Xtilde.T.dot(Wtilde)
    return z 

#Given the pre activation scores
#Creates a probability distribution for each class, by enforcing non-negativity and summation to 1.
#Then return the probability distribution
def probDistribution(z):
    pass

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    pass

if __name__ == "__main__":

    # Load data
    trainingImages = reshapeAndAugment(np.load("fashion_mnist_train_images.npy") / 255.0) # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = reshapeAndAugment(np.load("fashion_mnist_test_images.npy") / 255.0)  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")


    # Append a constant 1 term to each example to correspond to the bias terms
    # ...

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance, 
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    # ...

    # Train the model
    # Wtilde = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, alpha=.1)

    print(trainingImages.shape)
    # Visualize the vectors
    # ...