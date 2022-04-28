import numpy as np
import matplotlib.pyplot as plt
import random

#Given input images from raw dataset
#Reshaped and Apends 1 to correspond for the bias term from weights
#Return the Xtilde ((M+1)xN)
def reshapeAndAugmentX(input_images):
    ones = np.ones((1, input_images.shape[0]))
    input_images = input_images.T
    input_images = np.append(input_images, ones, axis=0)
    return input_images

def one_hot_encoding(y, no_classes):
    try:
        vector_labels = np.zeros((y.size,no_classes))
        vector_labels[np.arange(y.size),y] = 1
    except:
        vector_labels = None
    return vector_labels
 
#Given input data X and weights W
#Calculate the preactivation scores, i.e., Z = X^t * W
#Returns Z (NxK)
def preActivationScores(Xtilde, Wtilde):
    z = Xtilde.T.dot(Wtilde)
    return z

#Given the pre activation scores
#Creates a probability distribution for each class, by enforcing non-negativity and summation to 1.
#Then return the probability distribution (NxK)
def probDistribution(z):
    exp_z = np.exp(z)
    sum_z = np.sum(exp_z, axis=1).reshape(len(z),1)
    y_hat = exp_z/sum_z
    return y_hat

#Given input images X, weights W and corresponding Labels (vector form)
#Calculates the Cross Entropy loss
#Returns the cost of Cross Entropy loss
def fCE(Xtilde, Wtilde, y):
    z = preActivationScores(Xtilde, Wtilde)
    y_hat = probDistribution(z)
    y_hat = np.log(y_hat)
    inner_math = y*y_hat
    inner_math = np.sum(inner_math,axis=1)
    cost = np.mean(inner_math)*-1
    return cost

#Given input images Xtilde, Corresponding Labels (Vector form) Y, and normalized Predictions Y_hat
#Calculates Gradient of Cross Entropy Loss Function w.r.t Weights W
#Then return the gradient vector ((M+1)xK)
def gradeCE(Xtilde, Wtilde, Y, alpha=0.0):
    z = preActivationScores(Xtilde, Wtilde)
    Y_hat = probDistribution(z)
    # print(Xtilde.dot(Y_hat-Y))
    gradient_vector = Xtilde.dot(Y_hat- Y) 
    
    penalty = alpha*Wtilde
    penalty = penalty/(Xtilde.shape[1])
    gradient_vector += penalty

    return gradient_vector/len(Y)

#Given input images Xtilde, and batch_size
#Divides the training data into random batches based on batch_size
#Returns the list of randomized batches of training data [(M+1)xbacth_size, ...]
def prepare_training_data(Xtilde, Y, batch_size):
    no_batches = Xtilde.shape[1]/batch_size
    image_batches = np.hsplit(Xtilde, no_batches)
    label_batches = np.vsplit(Y, no_batches)
    return [image_batches, label_batches]

#Given the predictions Y^ and ground truth Y
#Calculates the percent correct accuracy
#Returns the percent correct accuracy value
def fPC (y, yhat):
    return np.count_nonzero(y==yhat)/y.size*100

# Given training and testing data, learning rate epsilon, batch size, and regularization strength alpha,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix Wtilde (785x10).
# Then return Wtilde.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon, batchSize, alpha):
    no_epochs = 10
    no_of_batches = (int)(trainingImages.shape[1]/batchSize)

    #initializing random weights ((M+1)xK)
    weights = 0.01*np.random.randn(trainingImages.shape[0]-1, trainingLabels.shape[1])
    w_ones = np.ones((1,trainingLabels.shape[1]))
    
    weights = np.append(weights, w_ones, axis=0)

    training_data = prepare_training_data(trainingImages, trainingLabels, batchSize)
    trainingImages = training_data[0]
    trainingLabels = training_data[1]

    random_batch_list = []
    
    counter = 0
    while(counter < no_of_batches):
        n = random.randint(1,no_of_batches)-1
        if(n not in random_batch_list):
            random_batch_list.append(n)
            counter += 1

    for epoch in range(no_epochs):
        mini_batch_counter = 1
        for random_batch_no in random_batch_list:
            
            training_images_batch = trainingImages[random_batch_no]
            training_labels_batch = trainingLabels[random_batch_no]

            gradient = gradeCE(training_images_batch, weights, training_labels_batch, alpha=alpha)
                        
            weights = weights - epsilon*gradient

            #Capturing fCE for last 20 batches
            cross_entropy_loss = fCE(training_images_batch, weights, training_labels_batch)
            print("Batch no: " + str(mini_batch_counter) + " Cross-Entropy Loss: " + str(cross_entropy_loss))
            mini_batch_counter += 1

    return weights


#Given the test images Xtilde and weights Wtilde
#Predicts the Categories of fahion
#Returns the predictions
def predict(Xtilde, Wtilde):
    z = preActivationScores(Xtilde, Wtilde)
    predictions = probDistribution(z)
    predictions_ = []
    for pred in predictions:
        pred[np.argmax(pred)] = 1
        pred = pred==1.0
        pred = pred*1.0
        predictions_.append(pred)
    predictions_ = np.array(predictions_)

    return predictions_


if __name__ == "__main__":

    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") / 255.0 # Normalizing by 255 helps accelerate training
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") / 255.0  # Normalizing by 255 helps accelerate training
    testingLabels = np.load("fashion_mnist_test_labels.npy")


    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = reshapeAndAugmentX(trainingImages)
    testingImages = reshapeAndAugmentX(testingImages)

    # Change from 0-9 labels to "one-hot" binary vector labels. For instance, 
    # if the label of some example is 3, then its y should be [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ]
    trainingLabels = one_hot_encoding(trainingLabels,10)
    testingLabels = one_hot_encoding(testingLabels,10)

    # Train the model
    Wtilde = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, alpha=.1)

    #Finding the accuracy of the model
    predictions = predict(testingImages, Wtilde)
    accuracy = fPC(predictions, testingLabels)
    print("\nPercent Correct Accuracy of the model on test set: " + str(accuracy))

    # Visualize the vectors
    fig = plt.figure(figsize=(10, 2))
    rows = 2
    columns = 5
    for i in range(0,10):
        image_ = Wtilde[:,i]
        image_ = np.delete(image_, -1)
        image_ = np.reshape(image_, (28,28))
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(image_)
        plt.axis('off')

    plt.show()
