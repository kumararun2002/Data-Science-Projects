from os import path
import numpy as np
import csv
import arff
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
    pred_Y = np.array(arff.load(open(predicted_test_Y_file_path, newline=''), 'rb')['data'])

    test_X_arff=arff.load(open(test_X_file_path,'r'))
    test_X=test_X_arff['data']
    test_X=np.asarray(test_X)

    #test_X=split_val(test_X,20)

    if pred_Y.shape != (len(test_X), 1):
        raise Exception("Output format is not proper")


def check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path):
    pred_Y = np.array(arff.load(open(predicted_test_Y_file_path, newline=''), 'rb')['data'])
    actual_Y = np.array(arff.load(open(actual_test_Y_file_path, newline=''), 'rb')['data'])
    #actual_Y=split_val(actual_Y,20)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(actual_Y, pred_Y)
    print("Accuracy", accuracy)
    return accuracy


def validate(test_X_file_path, actual_test_Y_file_path):
    predicted_test_Y_file_path = "predicted_test_Y_dt.arff"
    
    check_file_exits(predicted_test_Y_file_path)
    check_format(test_X_file_path, predicted_test_Y_file_path)
    check_accuracy(actual_test_Y_file_path, predicted_test_Y_file_path)