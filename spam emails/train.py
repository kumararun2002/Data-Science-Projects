import numpy as np
import csv
import pickle
import math
import json

def import_data():
    X = np.genfromtxt("train_X_nb.csv", dtype=np.str, delimiter='\n')
    Y = np.genfromtxt("train_Y_nb.csv", dtype=np.float64, delimiter='\n')
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

def save_model(class_wise_frequency_dict,prior_probabilities,class_wise_denominators):
    with open('CLASSWISE_FREQUENCY.json', 'w') as f:
        json.dump(class_wise_frequency_dict, f) 
    with open('PRIOR_PROB.json', 'w') as f:
        json.dump(prior_probabilities, f) 
    with open('CLASSWISE_DENO.json', 'w') as f:
        json.dump(class_wise_denominators, f) 
        
def compute_prior_probabilities(Y):
    Y1=np.copy(Y)
    dis=list(set(Y))
    dis.sort()
    dic={}
    for i in range(len(dis)):
        dic.update({dis[i]:1})
    for k in dic.keys():
        dic[k]=float(np.count_nonzero(Y==k)/len(Y))
    return Y1,dic
        
def class_wise_words_frequency_dict(X, Y):
    X1=np.copy(X)
    Y1=np.copy(Y)
    A=[]
    for i in range(np.shape(X)[0]):
        A.append(np.array(X[i].split(" ")))
    X=np.array(A)
    k=list(set(Y))
    dic={}
    for i in range(len(k)):
        temp={}
        for j in range(np.shape(X)[0]):
            if Y[j]==k[i]:
                for s in range(np.shape(X[j])[0]):
                    if X[j][s] in temp.keys():
                        temp[X[j][s]]=temp[X[j][s]]+1
                    else:
                        temp.update({X[j][s]:1})
        dic.update({k[i]:temp})
    return X1,Y1,dic

def get_class_wise_denominators_likelihood(X, Y ,alpha):
    X,Y,dic=class_wise_words_frequency_dict(X, Y)
    X1=np.copy(X)
    Y1=np.copy(Y)
    voca=0
    dis=[]
    likl={}
    for k in dic.keys():
        for s in dic[k].keys():
            dis.append(s)
    dis=set(dis)
    voca=len(dis)
    for k in dic.keys():
        temp=dic[k]
        temp1=list(temp.values())
        #print(temp1)
        val=np.sum(temp1)
        #print(val)
        likl.update({k:int(val+alpha*voca)})
    return X1,Y1,likl
       
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
    
def train_model(X,Y):
    for i in range(np.shape(X)[0]):
        X[i]=preprocessing(X[i])
    #print(X)
    X,Y,class_wise_frequency_dict=class_wise_words_frequency_dict(X, Y)
    #print(class_wise_frequency_dict.keys())
    Y,prior_probabilities = compute_prior_probabilities(Y)
    #print(prior_probabilities)
    X,Y,class_wise_denominators = get_class_wise_denominators_likelihood(X, Y,2.6)  ############
    #print(class_wise_denominators)
    return class_wise_frequency_dict,prior_probabilities,class_wise_denominators
    
if __name__=="__main__":
    X,Y=import_data()
    #X1,Y1=split_model(X,Y,20)
    X1=np.array(X)
    Y1=np.array(Y)
    #print(Y1)
    class_wise_frequency_dict,prior_probabilities,class_wise_denominators=train_model(X1,Y1)
    save_model(class_wise_frequency_dict,prior_probabilities,class_wise_denominators)