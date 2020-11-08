from os import path
import numpy as np
import csv
from train import Node
import math

def split_val(train_Y,validation_split_percent):
    val_Y=[]
    trno=math.floor(len(train_Y)*(100-validation_split_percent)/100)
    for i in range(trno,len(train_Y)):
        val_Y.append(train_Y[i])
    return np.array(val_Y)

def check_file_exits(predicted_test_Y_file_path):
    if not path.exists(predicted_test_Y_file_path):
        raise Exception("Couldn't find '" + predicted_test_Y_file_path +"' file")
        
def check_format(test_X_file_path, predicted_test_Y_file_path):
    pred_Y = []
    with open(predicted_test_Y_file_path, 'r') as file:
        reader = csv.reader(file)
        pred_Y = list(reader)
    pred_Y = np.array(pred_Y)

    test_X = np.genfromtxt(test_X_file_path, delimiter=',', \
        dtype=np.float64, skip_header=1)
    
    #test_X=split_val(test_X,20)

    if pred_Y.shape != (len(test_X), 1):
        raise Exception("Output format is not proper")


def check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path):
    pred_Y = np.genfromtxt(predicted_test_Y_file_path, delimiter=',', dtype=np.int)
    actual_Y = np.genfromtxt(actual_test_Y_file_path, delimiter=',', dtype=np.int)
    #actual_Y=split_val(actual_Y,20)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(actual_Y, pred_Y)
    print("Accuracy", accuracy)
    return accuracy


def validate(test_X_file_path, actual_test_Y_file_path):
    predicted_test_Y_file_path = "predicted_test_Y_de.csv"
    
    check_file_exits(predicted_test_Y_file_path)
    check_format(test_X_file_path, predicted_test_Y_file_path)
    check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path)