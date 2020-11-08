import numpy as np
import csv
import sys
import pickle
from validate import validate
from train import Node
import math

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_de.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model

def split_val(train_X,validation_split_percent):
    val_X=[]
    trno=math.floor(len(train_X)*(100-validation_split_percent)/100)
    for i in range(trno,len(train_X)):
        val_X.append(train_X[i])
    return np.array(val_X)

def predict_target_values(test_X, model):
    pred=[]
    for i in range(np.shape(test_X)[0]):
        node=model
        X=test_X[i]
        while node.left:
            if(X[node.feature_index]>=node.threshold):
                node=node.right
            else:
                node=node.left
        pred.append([node.predicted_class])
    return np.array(pred)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    #test_X=split_val(test_X,20)
    pred_Y = predict_target_values(test_X, model)
    #print(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 