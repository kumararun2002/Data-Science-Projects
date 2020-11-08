import numpy as np
import csv
import pickle
from sklearn import svm

def import_data():
    X = np.genfromtxt("train_X_svm.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_svm.csv", dtype=np.float64, delimiter=',')
    return X,Y

def save_model(weights):
    pickle.dump(weights, open('MODEL_FILE.sav', 'wb'))
    
def train_model(X,Y):
    weights=svm.SVC(C=1.5,kernel='linear')
    weights.fit(X,Y)
    return weights
    
if __name__=="__main__":
    X,Y=import_data()
    W=train_model(X,Y)
    save_model(W)
