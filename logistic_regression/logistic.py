import numpy as np
from random import random,sample

def sigmoid(t):
    return 1/(1 + np.exp(t))

def full_cost(theta,X,y):
    m = X.shape[0]
##    J = (1./m) * (-np.transpose(y).dot(np.log(sigmoid(X.dot(theta)))) -
##                  np.transpose(1-y).dot(np.log(sigmoid(X.dot(theta)))) )
##    grad = np.transpose((1./m)*np.transpose(sigmoid(X.dot(theta)) - y).dot(X))
    J_0 = np.sum(np.square(y - sigmoid(X.dot(theta))))
    return J_0

def make_batch_cost(batch_size):
    def batch_cost(theta,X,y):
        len_X = X.shape[0]
        index_sample = np.array(sample(range(len_X),batch_size))
        sub_X = X[index_sample]
        sub_y = y[index_sample]
        scale = len_X / sub_X.shape[0]
        return scale*np.sum(np.square(sub_y - sigmoid(sub_X.dot(theta))))
    return batch_cost

cost = full_cost

##def grad(theta,X,y):
##    m = X.shape[0]
##    out = np.transpose((1./m)*np.transpose(sigmoid(X.dot(theta)) - y).dot(X))
##    return out

def basis(m,n):
    out = []
    for i in range(m):
        row = []
        for j in range(n):
            b = np.zeros([m,n])
            b[i,j] = 1
            row.append(b)
        out.append(row)
    return out

def direc_deriv(f,x,dx,eps = 1e-5):
    return (f(x + eps*dx) - f(x))/eps

def dgrad(f,x,eps = 1e-5):
    m,n = x.shape
    out = 0*x
    B = basis(m,n)
    for i in range(m):
        for j in range(n):
            delta = direc_deriv(f,x,dx = B[i][j],eps = eps)
            out = out + delta*B[i][j]
    return out

def grad(theta,X,y,cost = None):
    if(cost is None):
        cost = full_cost
    cst = lambda theta: cost(theta,X,y)
    return dgrad(cst,theta)

def grad_desc(X,y,alpha = 0.1,method = 'iter',param = 10,cost = None):
    if(cost is None):
        cost = full_cost
    theta = initial_theta(shape = (X.shape[1],y.shape[1]))
    running = True
    if(method == 'iter'):
        n = 0
    while(running):
        theta = theta - alpha*grad(theta,X,y,cost = cost)
        if(method == 'iter'):
            n += 1
            if(n > param):
                running = False
    return(theta)

def initial_theta(shape):
    theta = np.zeros(shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            theta[i,j] = random()
    return theta

def predict(theta,X):
    return sigmoid(X.dot(theta))

# OR with bias
train_X = np.matrix([[1,0,0],[1,1,0],[1,0,1],[1,1,1]])
train_y = np.matrix([[0,1,1,1]]).T
def testing():
    batch = make_batch_cost(batch_size = 2)
    train_theta = grad_desc(train_X,train_y,param = 100,cost = batch)
    return train_theta
