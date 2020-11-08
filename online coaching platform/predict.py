import numpy as np
import sys
import csv
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

def import_data():
    X = np.genfromtxt("train_X_p1.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_p1.csv", dtype=np.float64, delimiter=',')
    return X,Y

def predict(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    test_X=np.delete(test_X,obj=0,axis=1)
    test_X=np.delete(test_X,obj=4,axis=1)
    test_X=np.delete(test_X,obj=8,axis=1)
    #test_X=np.delete(test_X,obj=7,axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    test_X=min_max_scaler.fit_transform(test_X)
    X,Y=import_data()
    X=np.delete(X,obj=0,axis=1)
    X=np.delete(X,obj=4,axis=1)
    X=np.delete(X,obj=8,axis=1)
    #X=np.delete(X,obj=7,axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    X=min_max_scaler.fit_transform(X)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X,Y)
    pred_Y=neigh.predict(test_X)
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open("predicted_test_Y_p1.csv", 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()
    

if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)