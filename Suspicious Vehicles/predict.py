import numpy as np
import csv
import sys
import pickle
from validate import validate
from train import Node

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_rf.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    models = pickle.load(open(model_file_path, 'rb'))
    return test_X, models

def predict_rf(test_X, model):
    pred=[]
    node=model
    X=test_X
    while node.left:
        if(X[node.feature_index]>=node.threshold):
            node=node.right
        else:
            node=node.left
    return node.predicted_class

def predict_using_bagging(models, test_X):
    test=[]
    for i in range(len(models)):
        test.append(predict_rf(test_X,models[i]))
    classes = sorted(list(set(test)))
    max_voted_class = -1
    max_votings = -1
    for c in classes:
        if(test.count(c) > max_votings):
            max_voted_class = c
            max_votings = test.count(c)
    return max_voted_class

def predict_target_values(test_X, models):
    pred_Y=[]
    for i in range(len(test_X)):
        pred_Y.append(predict_using_bagging(models, test_X[i]))
    return np.array(pred_Y)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, models = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, models)
    write_to_csv_file(pred_Y, "predicted_test_Y_rf.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_rf.csv") 