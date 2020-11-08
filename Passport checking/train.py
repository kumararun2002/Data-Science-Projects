import numpy as np
import csv
import pickle
import math

def import_data():
    X = np.genfromtxt("train_X_de.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_de.csv", dtype=np.float64, delimiter=',')
    return X,Y

def split_model(train_X,train_Y,validation_split_percent):
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
    return np.array(tra_X),np.array(tra_Y)

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None

def gini(Y):
    k=set(Y)
    k=list(k)
    Y=np.array(Y)
    s=len(Y)
    g=1
    for i in range(len(k)):
        g-=((np.count_nonzero(Y==k[i])/s)**2)
    return g
    
def calculate_gini_index(Y):
    Y=np.array(Y)
    m=np.shape(Y)[0]
    sum=0
    gin=0
    for i in range(m):
        sum+=len(Y[i])
    for i in range(m):
        gin+=(len(Y[i])/sum)*gini(Y[i])
    return gin

def get_best_split(X, Y):
    X = np.array(X)
    best_gini_index = 9999
    best_feature = 0
    best_threshold = 0
    for i in range(len(X[0])):
        thresholds = set(sorted(X[:, i]))
        for t in thresholds:
            left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
            if len(left_X) == 0 or len(right_X) == 0:
                continue
            gini_index = calculate_gini_index([left_Y, right_Y])
            if gini_index < best_gini_index:
                best_gini_index, best_feature, best_threshold = gini_index, i, t
    return best_feature, best_threshold

def split_data_set(data_X, data_Y, f, threshold):
   m=np.shape(data_X)[0]
   left_X=[]
   left_Y=[]
   right_X=[]
   right_Y=[]
   for i in range(m):
        if data_X[i][f]<threshold:
            left_X.append(data_X[i])
            left_Y.append(data_Y[i])
        else: 
            right_X.append(data_X[i])
            right_Y.append(data_Y[i])
   return left_X,left_Y,right_X,right_Y

def construct_tree(X, Y, max_depth, min_size, depth):
    classes = list(set(Y))
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    rootnode = Node(predicted_class, depth)
    if len(set(Y))==1:
        return rootnode
    if depth>=max_depth:
        return rootnode
    if len(Y)<=min_size:
        return rootnode
    best_feature,best_threshold=get_best_split(X, Y)
    if best_feature is None or best_threshold is None:
        return node
    left_X,left_Y,right_X,right_Y=split_data_set(X,Y, best_feature,best_threshold)
    rootnode.feature_index=best_feature
    rootnode.threshold=best_threshold
    rootnode.depth=depth
    rootnode.left=construct_tree(left_X, left_Y, max_depth, min_size, depth+1)
    rootnode.right=construct_tree(right_X, right_Y, max_depth, min_size, depth+1)
    return rootnode

def print_tree(node):
    if node.left!=None and node.right!=None:
        print("X",node.feature_index," ",node.threshold,sep="")
    if node.left!=None:
        print_tree(node.left)
    if node.right!=None:
        print_tree(node.right)


def save_model(weights):
    pickle.dump(weights, open('MODEL_FILE.sav', 'wb'))
    
def train_model(X,Y):
    node=construct_tree(X,Y,6,2,0)
    return node
    
if __name__=="__main__":
    X,Y=import_data()
    #X1,Y1=split_model(X,Y,20)
    W=train_model(X,Y)
    save_model(W)
