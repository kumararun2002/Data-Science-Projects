import numpy as np
import csv
import sys
import math


from validate import validate


"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_pr.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""
def import_data():
    X = np.genfromtxt("train_X_pr.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_pr.csv", dtype=np.float64, delimiter=',')
    return X,Y


def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights

def convert_given_cols_to_one_hot(X):
    m=(X.shape)[0]
    n=(X.shape)[1]
    dist=[]
    for i in range(n):
        temp=[]
        for j in range(m):
            if X[j][i] not in temp:
                temp.append(X[j][i])
        temp.sort()
        dist.append(temp)
    k=[]
    for i in range(len(dist)):
        temp=[]
        for j in range(len(dist[i])):
            temp.append(0)
        k.append(temp)
    ans=[]
    for i in range(m):
        temp2=[]
        for j in range(n):
            if j==3:
                ind=(dist[j]).index(X[i][j])
                temp=(k[j])[:]
                temp[ind]=1
                for z in range(len(temp)):
                    temp2.append(temp[z])
            else:
                temp2.append(X[i][j])
        ans.append(temp2)
    return np.array(ans),dist

def one_hot(X,dist):
    m=(X.shape)[0]
    n=(X.shape)[1]
    k=[]
    for i in range(len(dist)):
        temp=[]
        for j in range(len(dist[i])):
            temp.append(0)
        k.append(temp)
    ans=[]
    for i in range(m):
        temp2=[]
        for j in range(n):
            if j==3:
                ind=(dist[j]).index(X[i][j])
                temp=(k[j])[:]
                temp[ind]=1
                for z in range(len(temp)):
                    temp2.append(temp[z])
            else:
                temp2.append(X[i][j])
        ans.append(temp2)
    return np.array(ans)

def replace_null_values_with_mean(X):
    m=(X.shape)[0]
    n=(X.shape)[1]
    nor=[]
    arrsum=[]
    for i in range(n):
        count=0
        sum=0
        for j in range(m):
            if X[j][i]==X[j][i]:
                count+=1
                sum+=X[j][i]
        nor.append(count)
        arrsum.append(sum)
    for i in range(len(nor)):
        arrsum[i]/=nor[i]
    for i in range(n):
        for j in range(m):
            if X[j][i]!=X[j][i]:
                X[j][i]=arrsum[i]
    X=np.array(X)
    return X


def mean_normalize(X, column_indices):
    m=(X.shape)[0]
    n=(X.shape)[1]
    mean=[]
    minm=[]
    maxm=[]
    for i in range(n):
        row=[]
        for j in range(m):
            row.append(X[j][i])
        k=np.array(row)
        minm.append(np.amin(k))
        maxm.append(np.amax(k))
        mean.append(np.mean(k))
    for i in range(m):
        for j in range(n):
            if j in column_indices:
                X[i][j]=(X[i][j]-mean[j])/(maxm[j]-minm[j])
    return np.array(X)


def select_features(corr_mat, T1, T2):
    filfea=[]
    n=(corr_mat.shape)[1]
    for i in range(1,n):
        if abs(corr_mat[i][0])>T1:
            filfea.append(i)
    selfea=filfea[:]
    remfea=[]
    for i in range(len(filfea)):
        for j in range(i+1,len(filfea)):
            f1=filfea[i]
            f2=filfea[j]
            if f1 not in remfea and f2 not in remfea:
                if abs(corr_mat[f1][f2])>T2:
                    selfea.remove(f2)
                    remfea.append(f2)
    if 0 not in selfea:
        selfea.append(0)
    return np.array(selfea)


def get_stdmean(X):
    std=[]
    mean=[]
    m=(X.shape)[0]
    n=(X.shape)[1]
    for i in range(n):
        row=[]
        for j in range(m):
            row.append(X[j][i])
        k=np.array(row)
        std.append(np.std(k))
        mean.append(np.mean(k))
    return np.array(std),np.array(mean)


def get_correlation_matrix(X):
    m=(X.shape)[0]
    n=(X.shape)[1]
    cor=np.zeros((n,n))
    std,mean=get_stdmean(X)
    for i in range(n):
        for j in range(n):
            for k in range(m):
                cor[i][j]+=(X[k][i]-mean[i])*(X[k][j]-mean[j])/((m)*std[i]*std[j])
    return cor


def process_data(X1):
    X=replace_null_values_with_mean(X1)
    X,dist = convert_given_cols_to_one_hot(X)   
    m=(X.shape)[0]
    n=(X.shape)[1]
    col=[]
    for i in range(n):
        for j in range(m):
            if X[j][i]>1 or X[j][i]<0:
                col.append(i)
                break
    X2=mean_normalize(X,col)
    cor_mat= get_correlation_matrix(X2)
    fea = select_features(cor_mat, 0,0.6)
    count=0
    for i in range(n):
        if i not in fea:
            X2=np.delete(X2,i-count,axis=1)
            count+=1
    return X2,fea,dist


def sigmoid(Z):
    sig=(1.0/(1.0+np.exp(-Z)))
    return sig


def predict_target_values(test_X, weights):
    k=[]
    s=[]
    for i in range(2):
        b=weights[2][i]
        Z=np.dot(test_X,weights[i])+b
        A=sigmoid(Z)
        k.append(A)
    k=np.array(k)
    size=(k.shape)[1]
    for i in range(size):
        m=max(k[0][i],k[1][i])
        if m==k[0][i]:
            s.append([0])
        else:
            s.append([1])
    s=np.array(s)
    return s


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()
        
def save_model(weights,weights_file_name):
    with open(weights_file_name, 'w', newline='') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close() 


def split_val_data(test_X, validation_split_percent):
    """
    Returns best value of k which gives best accuracy
    """
    tes_X=[]
    trno=math.floor(len(test_X)*(validation_split_percent)/100)
    for i in range(trno,len(test_X)):
        tes_X.append(test_X[i])
    return np.array(tes_X)


def predict(test_X_file_path,fea,dist):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    #test_X = split_val_data(test_X,78)
    test_X = replace_null_values_with_mean(test_X)
    test_X=one_hot(test_X,dist)
    m=(test_X.shape)[0]
    n=(test_X.shape)[1]
    col=[]
    for i in range(n):
        for j in range(m):
            if test_X[j][i]>1 or test_X[j][i]<-1:
                col.append(i)
                break
    test_X=mean_normalize(test_X,col)
    count=0
    for i in range(n):
        if i not in fea:
            test_X=np.delete(test_X,i-count,axis=1)
            count+=1
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")
    
def split_train_data(train_X, train_Y, validation_split_percent):
    """
    Returns best value of k which gives best accuracy
    """
    tra_X=[]
    tra_Y=[]
    trno=math.floor(len(train_X)*(validation_split_percent)/100)
    for i in range(trno):
        tra_X.append(train_X[i])
        tra_Y.append(train_Y[i])
    return np.array(tra_X),np.array(tra_Y)


def compute_cost(X, Y, W, b ,Lambda):
    Z=np.dot(X,W)+b
    A=sigmoid(Z)
    np.where(A==1,0.99999,A)
    np.where(A==0,0.00001,A)
    cost = -1/Y.size * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) 
    cost += (Lambda/(2*Y.size))*np.sum(W*W)
    return cost


def compute_gradient_of_cost_function(X, Y, W, b ,l):
    Z=np.dot(X,W)+b
    A=sigmoid(Z)
    db=A-Y
    dw=np.matmul(X.T,db)+l*W
    return np.array(dw)/Y.size,np.sum(db)/Y.size


def optimize_weights_using_gradient_descent(X, Y, W, b,num, learning_rate , l):
    count=0
    while(1):
        count+=1
        prev_cost=compute_cost(X,Y,W,b,l)
        dw,db=compute_gradient_of_cost_function(X, Y, W,b ,l)
        W=np.subtract(W,learning_rate*dw)
        b=np.subtract(b,learning_rate*db)
        cost=compute_cost(X,Y,W,b,l)
        if abs(cost-prev_cost)<0.0000001:
            break
        num-=1
    return np.array(W),np.array(b) 


def get_train_data(X,Y,class_label):
    train_X=np.copy(X)
    train_Y=np.copy(Y)
    train_Y=np.where(Y==class_label,1,0)
    return train_X,train_Y


def train_model(X,Y,fea):
    k=[]
    s=[]
    b1=[]
    train_Y=Y.reshape(len(X),1)
    for i in range(2):
        X,Y=get_train_data(X,train_Y,i)
        W=np.zeros((X.shape[1],1))
        b=np.zeros((X.shape[0],1))
        W,b=optimize_weights_using_gradient_descent(X, Y, W,b,20000, 0.0068 ,0.097)
        for i in range(W.size):
            s.append(W[i][0])
        k.append(s)
        s=[]
        b1.append(b[0][0])
    for _ in range(len(fea)-2):
        b1.append(0)
    k.append(b1)
    k=np.array(k)
    return k


if __name__ == "__main__":
    X,Y=import_data()
    X,Y=split_train_data(X,Y,100)
    X,fea,dist=process_data(X)
    W=train_model(X,Y,fea)
    save_model(W,"WEIGHTS_FILE.csv")
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path,fea,dist)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv")