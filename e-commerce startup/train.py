import numpy as np
import csv

def import_data():
    X = np.genfromtxt("train_X_lr.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_lr.csv", dtype=np.float64, delimiter=',')
    return X,Y

def compute_gradient_of_cost_function(X, Y, W):
    m = len(X)
    Y_pred = np.dot(X, W)
    difference =  Y_pred - Y
    dW = (1/m) * (np.dot(difference.T, X))
    dW = dW.T
    return dW

def compute_cost(X,Y,W):
    Y1=np.dot(X,W)
    mse=np.sum(np.square(Y1-Y))
    val=mse/(2*len(X))
    return val

def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate):
    for z in range(1,num_iterations+1):
        gct=compute_gradient_of_cost_function(X, Y, W)
        W=np.subtract(W,learning_rate*gct)
        cost=compute_cost(X,Y,W)
        if z%10000000==0:
            print(z, cost)
    return W 

def train_model(X,Y):
    X=np.insert(X,0,1,axis=1)
    Y=Y.reshape(len(X),1)
    W=np.zeros((X.shape[1],1))
    W=optimize_weights_using_gradient_descent(X, Y, W, 200000000, 0.000213)
    return W

def save_model(weights,weights_file_name):
    with open(weights_file_name,'w')  as w_file:
        wr=csv.writer(w_file)
        wr.writerows(weights)
        w_file.close()

if __name__=="__main__":
    X,Y=import_data()
    W=train_model(X,Y)
    save_model(W,"WEIGHTS_FILE.csv")

