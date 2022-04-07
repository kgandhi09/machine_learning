import numpy as np
from homework3_Gandhi import softmaxRegression,reshapeAndAugmentX,one_hot_encoding, predict
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
    
    training_input = np.stack((gender, passenger_class,sib_sp), axis=0)

    return [training_input.T, survived]

def get_pass_id(data_path):
    data = pd.read_csv(data_path)
    pass_id = data['PassengerId'].tolist()
    return pass_id

if __name__ == "__main__":
    training_data = process_training_data("titanic_train.csv")
    training_input = reshapeAndAugmentX(training_data[0])
    training_labels = one_hot_encoding(training_data[1],2) #[didn't survive, survived] -> classes

    testing_data = process_training_data("titanic_test.csv")
    testing_input = reshapeAndAugmentX(testing_data[0])
    testing_labels = one_hot_encoding(testing_data[1],2)

    Wtilde = softmaxRegression(training_input, training_labels, testing_input, testing_labels, epsilon=0.1, batchSize=33, alpha=.1)
    predictions = predict(testing_input, Wtilde)
    survived_test = []

    pass_id_test = get_pass_id("titanic_test.csv")

    for pred in predictions:
        if pred[0] == 1:
            survived_test.append(0)
        else:
            survived_test.append(1)


    df = pd.DataFrame({'PassengerId':pass_id_test, 'Survived':survived_test})
    df.to_csv('titanic_predictions.csv', index = False)