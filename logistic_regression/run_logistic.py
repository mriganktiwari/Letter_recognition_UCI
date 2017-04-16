import numpy as np
from string import ascii_uppercase
from logistic import predict,grad_desc,make_batch_cost
from copy import deepcopy

directory = "/Users/jacobmaibach/Documents/School/Spring 2017/Machine Learning I/Project/Data"
data = np.genfromtxt(directory + "/letter_recognition_raw.txt",dtype = str,delimiter = ",")

inverse_letter_lookup = {c:i for i,c in enumerate(ascii_uppercase)}

letter_data = data[:,0]
altered_data = deepcopy(data)
altered_data[:,0] = [1]*data.shape[0]
# X = np.float64(data[:,1:]) # without bias
X = np.float64(altered_data)
y = np.zeros(shape = (X.shape[0],26))
for i in range(len(letter_data)):
    j = inverse_letter_lookup[letter_data[i]]
    y[i,j] = 1

def run_logistic(iterations = 100,batch_size = 10,alpha = 0.1):
    batch_cost = make_batch_cost(batch_size)
    theta = grad_desc(X,y,alpha = alpha,param = iterations,cost = batch_cost)
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



