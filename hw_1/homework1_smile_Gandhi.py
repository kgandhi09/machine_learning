import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.count_nonzero(y==yhat)/y.size*100

def measureAccuracyOfPredictors (predictors, X, y):
    sum = np.zeros(y.shape)
    for predictor in predictors:
        r1 = predictor[0]
        c1 = predictor[1]
        r2 = predictor[2]
        c2 = predictor[3]
        predictions = X[:,r1,c1] - X[:,r2,c2] > 0
        predictions = predictions*1
        sum += predictions

    ensemble = sum/len(predictors)
    ensemble_predictions = ensemble>0.5
    ensemble_predictions = ensemble_predictions*1
    return fPC(y, ensemble_predictions)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    show = True
    best = 0
    best_predictor = [0,0,0,0]
    predictors = []

    #Comparing one pixel values with every other
    for i in range(6):
        for r1 in range(24):
            for c1 in range(24):
                for r2 in range(24):
                    for c2 in range(24):
                        if((r1,c1) != (r2,c2)):
                            acc = measureAccuracyOfPredictors(predictors+[[r1,c1,r2,c2]], trainingFaces, trainingLabels)
                            if(acc > best):
                                best = acc
                                best_predictor = [r1,c1,r2,c2]

        predictors.append(best_predictor)
    
    visualize(show, predictors)
    return predictors

def visualize(show, predictors):
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        for i in range(len(predictors)):
            r1 = predictors[i][0]
            c1 = predictors[i][1]
            r2 = predictors[i][2]
            c2 = predictors[i][3]
            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            # Display the merged result
        plt.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    no_examples = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    for n in no_examples:
        predictors = stepwiseRegression(trainingFaces[:n], trainingLabels[:n], testingFaces, testingLabels)
        train_acc = measureAccuracyOfPredictors(predictors, trainingFaces[:n], trainingLabels[:n])
        test_acc = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        print(str(n) + " Training Accuracy: " + str(train_acc) + " Testing Accuracy: " + str(test_acc))