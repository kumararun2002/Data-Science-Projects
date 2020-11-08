import numpy as np
import csv
import sys
import arff
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import math
from train import Node

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training
Writes the predicted values to the file named "predicted_test_Y_dt.arff". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a proper implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X_arff=arff.load(open(test_X_file_path,'r'))
    test_X=test_X_arff['data']
    test_X=np.asarray(test_X)
    model = pickle.load(open(model_file_path, 'rb'))
    ind = pickle.load(open('ind.sav', 'rb'))
    categories = pickle.load(open('categories.sav', 'rb'))
    return test_X, model, categories,ind

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

def write_to_arff_file(pred_Y, predicted_Y_file_name):
    test_X_arff=arff.load(open(test_X_file_path,'r'))
    arff_data={
        'data':pred_Y, 
        'relation':test_X_arff['relation'], 'description':'', 
        'attributes':[('class',['True','False'])]
        }
    with open('./predicted_test_Y_dt.arff','w') as arff_file:
        arff.dump(arff_data,arff_file)
       
def preprocess(X,categories,ind):
    X=np.delete(X,ind,axis=1)
    imp = SimpleImputer(missing_values=None, strategy='most_frequent')
    imp.fit(X)
    X=imp.transform(X)
    enc=OrdinalEncoder(categories=categories)
    enc.fit(X)
    X=enc.transform(X)
    #print(X[0])
    return X


def predict(test_X_file_path):
    test_X, model ,category,ind = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    #test_X=split_val(test_X,20)
    test_X=preprocess(test_X,category,ind)
    pred_Y = predict_target_values(test_X, model)
    write_to_arff_file(pred_Y,  "predicted_test_Y_dt.arff")
    

if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_dt.arff") 