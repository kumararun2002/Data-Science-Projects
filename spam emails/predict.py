import numpy as np
import csv
import sys
import json
import math

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    return test_X

def import_model(model_file_path):
    with open(model_file_path) as f:
        model = json.load(f)
    return model

def split_val(train_X,validation_split_percent):
    val_X=[]
    trno=math.floor(len(train_X)*(100-validation_split_percent)/100)
    for i in range(trno,len(train_X)):
        val_X.append(train_X[i])
    return np.array(val_X)

def compute_likelihood(test_X, c,class_wise_frequency_dict, class_wise_denominators,alpha):
    freq=class_wise_frequency_dict[c]
    for k in freq.keys():
        freq[k]=freq[k]+alpha
    deno=class_wise_denominators[c]
    test_X=test_X.split(" ")
    likl=0
    for i in range(len(test_X)):
        if test_X[i] in freq.keys():
            likl+=math.log(freq[test_X[i]]/deno)
        else:
            likl+=math.log(alpha/deno)
    return likl

def predict_target_values(X,class_wise_frequency_dict,prior_probabilities,class_wise_denominators,alpha):
    classes=list(class_wise_frequency_dict.keys())
    classes.sort()
    ans=[]
    for i in range(np.shape(X)[0]):
        test_X=X[i]
        pclass=classes[0]
        comlik=compute_likelihood(test_X, pclass, class_wise_frequency_dict, class_wise_denominators,alpha) 
        #print(comlik,pclass)
        for cl in classes:
            #print(compute_likelihood(test_X, cl, class_wise_frequency_dict, class_wise_denominators),cl)
            if comlik<compute_likelihood(test_X, cl, class_wise_frequency_dict, class_wise_denominators,alpha):
                comlik=compute_likelihood(test_X, cl, class_wise_frequency_dict, class_wise_denominators,alpha)
                pclass=cl
        ans.append(pclass)
    return np.array(ans)

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()
        
def preprocessing(s):
    k=""
    for i in range(len(s)):
        if s[i].isalpha():
            break
    while i<len(s):
        if s[i].isalpha():
            k+=s[i].lower()
        if s[i]==" ":
            if k[len(k)-1]!=" ":
                k+=s[i]
        i=i+1
    i=len(k)-1
    while i>=0:
        if k[i].isalpha():
            break
        i=i-1
    k=k[:i+1]
    return k

def predict(test_X_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    test_X= import_data(test_X_file_path)
    #test_X=split_val(test_X,20)
    class_wise_frequency_dict=import_model("CLASSWISE_FREQUENCY.json")
    prior_probabilities=import_model("PRIOR_PROB.json")
    class_wise_denominators=import_model("CLASSWISE_DENO.json")
    for i in range(np.shape(test_X)[0]):
        test_X[i]=preprocessing(test_X[i])
    pred_Y = predict_target_values(test_X,class_wise_frequency_dict,prior_probabilities,class_wise_denominators,2.7)   #############
    #print(np.shape(pred_Y),np.shape(test_X))
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 