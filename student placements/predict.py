import numpy as np
import csv
import sys
import math

def import_train_data():
    train = np.genfromtxt("train_X_re.csv", dtype=np.float64, delimiter=',', skip_header=1)
    val = np.genfromtxt("train_Y_re.csv", dtype=np.float64, delimiter=',')
    return train,val

def compute_ln_norm_distance(vector1, vector2, n):
    sum=0
    for i in range(len(vector1)):
        k=abs(vector1[i]-vector2[i])
        k=k**n
        sum+=k
    return sum**(1/n)

def find_k_nearest_neighbors(train_X, test_example, k, n_in_ln_norm_distance):
    nearneb=[]
    ind=[]
    for i in range(len(train_X)):
        kdis=compute_ln_norm_distance(train_X[i],test_example,n_in_ln_norm_distance)
        nearneb.append((kdis,i))
    nearneb.sort()
    for i in range(k):
        ind.append(nearneb[i][1])
    return np.array(ind)

def classify_points_using_knn(train_X,train_Y, test_X, k, n_in_ln_norm_distance):
    indic=[]
    for i in range(len(test_X)):
        indic.append((find_k_nearest_neighbors(train_X, test_X[i], k, n_in_ln_norm_distance)))
    ans=[]
    for i in range(len(indic)):
        s=0
        for j in range(len(indic[i])):
            s+=train_Y[indic[i][j]]
        s/=len(indic[i])
        ans.append(s)
    return np.array(ans)

def calculate_mse(predicted_Y, actual_Y):
    """
    Arguments:
    predicted_Y - 1D array of predicted target values
    actual_Y - 1D array of true values 

    Returns:
    accuracy
    """
    #count=0
    s=0
    for i in range(len(actual_Y)):
        s+=(predicted_Y[i]-actual_Y[i])**2
    return s/len(actual_Y)

def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n_in_ln_norm_distance):
    """
    Returns best value of k which gives best accuracy
    """
    tra_X=[]
    tra_Y=[]
    val_X=[]
    val_Y=[]
    trno=math.floor(len(train_X)*(100-validation_split_percent)/100)
    for i in range(trno):
        tra_X.append(train_X[i])
        tra_Y.append(train_Y[i])
    for i in range(trno,len(train_X)):
        val_X.append(train_X[i])
        val_Y.append(train_Y[i])
    paacu=[]
    for i in range(1,trno+1):
        pts=classify_points_using_knn(tra_X, tra_Y, val_X,i, n_in_ln_norm_distance)
        accu=calculate_mse(pts, val_Y)
        paacu.append((accu,i))
    paacu.sort()
    #print(paacu)
    return paacu[0][1]


from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X


def predict_target_values(test_X):
    # Write your code to Predict Target Variables
    train,val=import_train_data()
    k=get_best_k_using_validation_set(train, val, 80,n_in_ln_norm_distance=1)
    clpts=classify_points_using_knn(train, val, test_X, k, n_in_ln_norm_distance=1)
    return clpts

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_re.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_re.csv") 