import numpy as np
import csv
import pickle
import math
import arff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def import_data():
    data_X = arff.load(open('train_X_dt.arff', 'r'))
    data_Y = arff.load(open('train_Y_dt.arff', 'r'))
    X=data_X['data']
    X=np.asarray(X)
    Y=data_Y['data']
    Y=np.asarray(Y)
    category=[]
    for i in range(np.shape(data_X['attributes'])[0]):
        category.append(data_X['attributes'][i][1])
    return X,Y,category

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
    if len(classes)==0:
        return None
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

def save_model(weights):
    pickle.dump(weights, open('MODEL_FILE.sav', 'wb'))
    
def train_model(X,Y):
    node=construct_tree(X,Y,50,10,0) #############################
    return node

def preprocess(X,category):
    ind=[]
    for i in range(np.shape(X)[1]):
        count=0
        for j in range(np.shape(X)[0]):
            if X[j][i]==None:
                count+=1
        if count>(np.shape(X)[0])/2:  ##################################
            ind.append(i)
    X=np.delete(X,ind,axis=1)
    category=np.delete(category,ind,axis=0)
    categories=[]
    for i in range(np.shape(category)[0]):
        categories.append(category[i])
    pickle.dump(categories, open('categories.sav', 'wb'))
    pickle.dump(ind, open('ind.sav', 'wb'))
    imp = SimpleImputer(missing_values=None, strategy='most_frequent')
    imp.fit(X)
    X=imp.transform(X)
    enc=OrdinalEncoder(categories=categories)
    enc.fit(X)
    X=enc.transform(X)
    #print(X[0])
    return X
    
if __name__=="__main__":
    X,Y1,category=import_data()
    Y=[]
    for i in range(np.shape(Y1)[0]):
        Y.append(Y1[i][0])
    #X,Y=split_model(X,Y,20)
    X=preprocess(X,category)
    W=train_model(X,Y)
    save_model(W)
