## Jacob Maibach, 2017
## Logistic regression applied to Letter Recognition Dataset

import numpy as np
from string import ascii_uppercase
from logistic import predict,grad_desc,make_batch_cost,p_grad_desc
from copy import deepcopy
from statistics import stdev,mean

directory = "/Users/jacobmaibach/Documents/School/Spring 2017/Machine Learning I/Project/Data"
data = np.genfromtxt(directory + "/letter_recognition_raw.txt",dtype = str,delimiter = ",")

inverse_letter_lookup = {c:i for i,c in enumerate(ascii_uppercase)}

letter_data = data[:,0]
numeric_data = np.float64(data[:,1:])

# normalization
data_variables = numeric_data.transpose()
data_sd = [stdev(var) for var in data_variables]
data_mean = [mean(var) for var in data_variables]

normalized = (numeric_data - data_mean)/data_sd

X = np.insert(normalized,0,values = 1,axis = 1)

y = np.zeros(shape = (X.shape[0],26))
for i in range(len(letter_data)):
    j = inverse_letter_lookup[letter_data[i]]
    y[i,j] = 1

def run_logistic(iterations = 100,batch_size = 10,alpha = 0.1):
    batch_cost = make_batch_cost(batch_size)
    theta = grad_desc(X,y,alpha = alpha,param = iterations,cost = batch_cost)
    return theta

def run_logistic_p(iterations = 100, batch_size = 20, alpha = 0.1,beta = 0.1):
    batch_cost = make_batch_cost(batch_size)
    theta = p_grad_desc(X,y,alpha = alpha,beta = beta,param = iterations,cost = batch_cost)
    return theta

def max_index(v):
    max_val = v[0]
    max_ind = 0
    for i in range(1,len(v)):
        if(v[i] > max_val):
            max_val = v[i]
            max_ind = i
    return(max_ind)

def letter_prediction(theta,X,i):
    sub_X = X[i:i+1]
    p = predict(theta,sub_X)[0]
    j = max_index(p)
    return ascii_uppercase[j]

def full_letter_prediction(theta,X):
    s = ''.join(letter_prediction(theta,X,i) for i in range(len(X)))
    counts = {c:s.count(c) for c in ascii_uppercase}
    return counts

def sum_prob_prediction(theta,X):
    pred_len = 26
    tot = np.zeros(shape = [pred_len])
    for i in range(len(X)):
        tot += predict(theta,X[i:i+1])[0] # prediction of X_i
    out = tot/len(X)
    return {ascii_uppercase[i]:out[i] for i in range(pred_len)}

def view_prob_prediction(theta,X,num_digits = 3):
    tot = sum_prob_prediction(theta,X)
    scale = 10**num_digits
    for c in tot:
        print(c,'0.{0}'.format(int(scale*tot[c])))

def accuracy(theta,X,y):
    tot_correct = 0
    for i in range(X.shape[0]):
        p = predict(theta,X[i:i+1])[0]
        tot_correct += (max_index(p) == max_index(y[i]))
    return tot_correct/X.shape[0]


