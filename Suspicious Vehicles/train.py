import numpy as np
import csv
import pickle
import math
import random

def import_data():
    X = np.genfromtxt("train_X_rf.csv", dtype=np.float64, delimiter=',', skip_header=1)
    Y = np.genfromtxt("train_Y_rf.csv", dtype=np.float64, delimiter=',')
    return X,Y

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

def get_best_split(X, Y,features):
    X = np.array(X)
    best_gini_index = 9999
    best_feature = 0
    best_threshold = 0
    for i in features:
        thresholds = set(sorted(X[:, i]))
        for t in thresholds:
            left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
            if len(left_X) == 0 or len(right_X) == 0:
                continue
            gini_index = calculate_gini_index([left_Y, right_Y])
            if gini_index < best_gini_index:
                best_gini_index, best_feature, best_threshold = gini_index, i, t
    return best_feature, best_threshold

def get_best_split_random(X, Y,num_features):
    feature_indices=[]
    total_num_features = len(X[0])
    while len(feature_indices) < num_features:
        index = random.randint(0,total_num_features-1)
        if index not in feature_indices:
            feature_indices.append(index)
    return get_best_split(X, Y, feature_indices)

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

def construct_tree(X, Y, depth):
    classes = list(set(Y))
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    rootnode = Node(predicted_class, depth)
    if len(set(Y))==1:
        return rootnode
    best_feature,best_threshold=get_best_split_random(X, Y,8)
    if best_feature is None or best_threshold is None:
        return node
    left_X,left_Y,right_X,right_Y=split_data_set(X,Y, best_feature,best_threshold)
    rootnode.feature_index=best_feature
    rootnode.threshold=best_threshold
    rootnode.depth=depth
    rootnode.left=construct_tree(left_X, left_Y, depth+1)
    rootnode.right=construct_tree(right_X, right_Y,depth+1)
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
    node=construct_tree(X,Y,0)
    return node

def predict(test_X, model):
    pred=[]
    node=model
    X=test_X
    while node.left:
        if(X[node.feature_index]>=node.threshold):
            node=node.right
        else:
            node=node.left
    return node.predicted_class

def get_out_of_bag_error(models,X,Y, bootstrap_samples_X,bootstrap_samples_Y):
    m=np.shape(X)[0]
    n=np.shape(X)[1]
    berr=0
    tcnt=0
    for i in range(m):
        err=0
        count=0
        for j in range(len(bootstrap_samples_X)):
            #print(bootstrap_samples_X[j])
            if list(X[i]) not in list(bootstrap_samples_X[j]):
                count+=1
                k=predict(X[i],models[j])
                if (k-Y[i])!=0:
                    err+=1
        if count>0:
            berr+=(err/float(count))   
            tcnt+=1
    berr/=float(tcnt)
    return berr

def get_bootstrap_samples(X,Y, num_bootstrap_samples):
    m=np.shape(X)[0]
    btsr=[]
    btsr_Y=[]
    for _ in range(num_bootstrap_samples):
        temp=[]
        temp1=[]
        for i in range(m):
            k=random.randint(0,m-1)
            temp.append(list(X[k]))
            #print(Y[k])
            temp1.append(Y[k])
        btsr.append(temp)
        btsr_Y.append(temp1)
    return btsr,btsr_Y      
    
if __name__=="__main__":
    X,Y=import_data()
    error=10000000.0
    bt_nums=30
    while bt_nums<=30:
        #random.seed()
        models=[]
        btsp_X,btsp_Y = get_bootstrap_samples(X,Y, bt_nums)
        for i in range(len(btsp_X)):
            models.append(train_model(np.array(btsp_X[i]),np.array(btsp_Y[i])))
        oob = get_out_of_bag_error(models,X,Y,btsp_X,btsp_Y)
        #print(oob,bt_nums)
        if oob<=error:
            error=oob
            W=models
        bt_nums+=1
    #print(error)
    save_model(W)
