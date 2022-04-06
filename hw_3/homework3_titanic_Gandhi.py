import numpy as np
import csv
from homework3_Gandhi import softmaxRegression,reshapeAndAugmentX,one_hot_encoding
import pandas as pd

#Helper function for processign training data
def separate_string(string_):
    x = string_.split(",")
    return x

#Given the training data csv path for titanic challenge
#Loads and processes the csv to get training data and labels
#Returns Training Input (X) and Corresponding Labels (Y)
def process_training_data(data_path):
    data = pd.read_csv(data_path)
    try:
        survived=np.array(data['Survived'].tolist())
    except KeyError:
        survived = None
    passenger_class = np.array(data['Pclass'].tolist())
    gender_raw = data['Sex'].tolist()
    gender = []
    [gender.append(int(sex=='female')) for sex in gender_raw]
    gender = np.array(gender)
    age = np.array(data['Age'].tolist())
    sib_sp = np.array(data['SibSp'].tolist())
    parch = np.array(data['Parch'])
    
    training_input = np.stack((passenger_class,gender,age,sib_sp,parch), axis=0)

    return [training_input.T, survived]

if __name__ == "__main__":
    training_data = process_training_data("titanic_train.csv")
    training_input = reshapeAndAugmentX(training_data[0])
    training_labels = one_hot_encoding(training_data[1],2) #[didn't survive, survived] -> classes

    testing_data = process_training_data("titanic_test.csv")
    testing_input = reshapeAndAugmentX(testing_data[0])
    testing_labels = one_hot_encoding(testing_data[1],2)

    Wtilde = softmaxRegression(training_input, training_labels, testing_input, testing_labels, epsilon=0.1, batchSize=33, alpha=.1)

    print(Wtilde.shape)
