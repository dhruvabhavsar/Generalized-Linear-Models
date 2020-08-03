#!/usr/local/bin/python3
# Submitted by: Dhruva Bhavsar, Username:dbhavsar
# Programming Project 3


import numpy as np
import matplotlib.pyplot as plt
import random
import sys


# Function to calculate sigmoid
def sigmoid(x):
   return 1/(1+np.exp(-x))

# Split a dataset into k folds
def cross_validation_split(dataset, dataset1,folds):
    dataset_split = []
    dataset_split1=[]
    dataset_copy = list(dataset)
    dataset_copy1=list(dataset1)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        fold1=list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            fold1.append(dataset_copy1.pop(index))
        dataset_split.append(fold)
        dataset_split1.append(fold1)
    d_split=np.array(dataset_split)
    d_split1=np.array(dataset_split1)
    return d_split,d_split1   

# Main function for all generalized linear models 
def GLM(phi,t,phi_test,t_test,model,phi_j,s):
    
    # Initialize w to 0
    w=np.zeros((phi.shape[1],1))
    t=t[:,np.newaxis]
    t_test=t_test[:,np.newaxis]
    conv=1

    niter=0
    
    # Calculate the value of w until convergence
    while conv >= 10**-3 and niter<50:
        
        wn=w

        a=np.dot(phi,w)
        y=np.zeros(t.shape)
        d=np.zeros(t.shape)
        r=np.zeros(t.shape)
        
        # Calculate d and r based for the different models
        if(model=='Logistic'):
            y=sigmoid(a)
            d=t-y
            for i in range(0,phi.shape[0]):
                r[i][0]=y[i][0]*(1-y[i][0])
        elif(model=='Poisson'):
            y=np.exp(a)
            d=t-y
            for i in range(0,phi.shape[0]):
                r[i][0]=y[i][0]
        elif(model=='Ordinal'):
            for i in range(0,a.shape[0]):
                for j in range(0,phi_j.shape[0]):
                    if(t[i]==j):
                        yij=sigmoid(s*(phi_j[j]-a[i]))
                        yij1=sigmoid(s*(phi_j[j-1]-a[i]))
                        d[i]=yij+yij1-1
                        r[i]=(s**2)*(yij*(1-yij)+yij1*(1-yij1))
            
        l=np.dot(phi.T,d)
        
        R=r*np.identity(phi.shape[0])
        
        # Calculate g
        g=l-alpha*w
        N=np.dot(phi.T,R)
        
        # Calculate H
        h=-np.dot(N,phi)-alpha*np.eye(phi.shape[1])
#        print(np.linalg.cond(h))
        if(np.linalg.cond(h)==np.inf or np.linalg.cond(h)>1.0169322392495714e+19):
            niter+=1
            continue
        # Calculate w ← w − H−1 (Newton's method)
        w = np.subtract(w,(np.dot(np.linalg.inv(h),g)))

        if (np.linalg.norm(wn))==0:
            continue
        conv=(np.linalg.norm(w-wn))/np.linalg.norm(wn)
        niter+=1
        
    
    # Make predictions based on model
    if(model=='Logistic'):
        return predict_logistic(phi_test,t_test,w)
    elif(model=='Poisson'):
        return predict_poisson(phi_test,t_test,w)
    elif(model=='Ordinal'):
        return predict_ordinal(phi_test,t_test,w,phi_j,s)
    
# Function to predict and find error for Logistic Regression    
def predict_logistic(phi_test,t_test,w):
    pred=[]
    for j in range(0,phi_test.shape[0]):
        if sigmoid(np.dot(w.T,phi_test[j]))>= 0.5:
            pred.append(1)
        else:
            pred.append(0)
    
    error=0
    for k in range(0,t_test.shape[0]):
        error+=abs(t_test[k][0]-pred[k])
    
    return error/t_test.shape[0]    

# Function to predict and find error for Poisson Regression
def predict_poisson(phi_test,t_test,w):
    pred=[]
    for j in range(0,phi_test.shape[0]):
        pred.extend(np.exp(np.dot(w.T,phi_test[j])))
       
    error=0
    for k in range(0,t_test.shape[0]):
        error+=abs(t_test[k][0]-pred[k])
    
    return (error/t_test.shape[0])

# Function to predict and find error for Ordinal Regression
def predict_ordinal(phi_test,t_test,w,phi_j,s):
    pred=[]
    for i in range(0,phi_test.shape[0]):
        a=np.dot(w.T,phi_test[i])
        pj={}
        for j in range(1,phi_j.shape[0]):
            yij=sigmoid(s*(phi_j[j]-a))
            yij1=sigmoid(s*(phi_j[j-1]-a))
            pj[j]=yij-yij1
        k=max(pj.keys(), key=(lambda k: pj[k]))
        pred.append(k)       

    error=0
    acc=0
    for k in range(0,t_test.shape[0]):
        if(t_test[k][0]==pred[k]):
            acc+=1
        else:
            acc+=0
        error+=abs(t_test[k][0]-pred[k])
    
    return (error/t_test.shape[0])


if __name__ == "__main__":
    
   # Initialize dataset files and model 
   file1=sys.argv[1]
   file2=sys.argv[2]
   model=sys.argv[3]
   
   # Read data from the given dataset files 
   data=np.loadtxt('pp3data/'+file1,delimiter=',')
   
   label=np.loadtxt('pp3data/'+file2,delimiter=',')
   
   # Initialize values    
   phi_j=np.array([-np.inf,-2,-1,0,1,np.inf])
   K=5
   s=1
#   alpha=10

   # Add intercept to the dataset
   a=np.ones((data.shape[0],1))

   b=np.hstack((a,data))

   acc1={}
   
   # Dividing the dataset using cross validation
   dataset,dataset1=cross_validation_split(b,label,5)
   for j in range(0,5):
        testset=[]
        testsetR=[]
        trainset=[]
        trainsetR=[]
        a=[]
        for i in range(0,5):
            if(i!=j):
                trainset.extend(dataset[i])
                trainsetR.extend(dataset1[i])
            else:
                testset.extend(dataset[i])
                testsetR.extend(dataset1[i])  
        train=np.array(trainset)
        train_label=np.array(trainsetR)
        test=np.array(testset)
        test_label=np.array(testsetR)
        # Calculate the error for alphas from 0 to 100
        for alpha in range(0,100):
            
            error=GLM(train,train_label,test,test_label,model,phi_j,s)
            if alpha in acc1:
               acc1[alpha].append(error)
            else:
               acc1[alpha]=[error]
#   print(acc1) 
   mean={}
   mean1=[]

   size=[]
   for n in acc1:
       mean[n]=np.mean(acc1[n])
       mean1.append(np.mean(acc1[n]))

       size.append(n)
#   print(mean1)
       
   # Selecting alpha with the lowest mean absolute value
   k=min(mean.keys(), key=(lambda k: mean[k]))
   print("Alpha =",k)
   plt.xlabel('Alpha')
   plt.ylabel('Average Error')
   plt.plot(size,mean1)
   plt.show()