import numpy as np
from string import ascii_uppercase
from logistic import predict,grad_desc,make_batch_cost

directory = "/Users/jacobmaibach/Documents/School/Spring 2017/Machine Learning I/Project/Data"
data = np.genfromtxt(directory + "/letter_recognition_raw.txt",dtype = str,delimiter = ",")

inverse_letter_lookup = {c:i for i,c in enumerate(ascii_uppercase)}

letter_data = data[:,0]
X = np.float64(data[:,1:])
y = np.zeros(shape = (X.shape[0],26))
for i in range(len(letter_data)):
    j = inverse_letter_lookup[letter_data[i]]
    y[i,j] = 1

def run(iterations = 100,batch_size = 100):
    batch_cost = make_batch_cost(batch_size)
    theta = grad_desc(X,y,param = iterations,cost = batch_cost)
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
    return ascii_letters[j]



