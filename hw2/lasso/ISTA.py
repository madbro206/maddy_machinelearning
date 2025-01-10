from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    #follow steps for Iterative Shrinkage Thresholding Algorithm for Lasso
    n,d = X.shape
    
    """
    #wrong algorithm :/
    #bias (calculated bias on input weight)
    b = (1/n) * np.sum(y-np.dot(X,weight))
    
    for k in range(1,d):
        #calculate a_k
        #kth column of X
        X_k = X[:,k]
        a_k = 2 * np.sum(X_k**2)

        #calculate c_k
        #remove kth column/entry from X and weight
        X_no_k = np.delete(X, k, axis=1)
        weight_no_k = np.delete(weight, k)
        c_k = 2* np.dot(X_k, y-(b + np.dot(X_no_k, weight_no_k)))
        
        #update weights
        if c_k < -_lambda:
            weight[k] = (c_k + _lambda)/a_k
        elif c_k > _lambda:
            weight[k] = (c_k - _lambda)/a_k
        else:
            weight[k]=0
 
    """
    b_ = bias - 2 * eta * np.sum(np.dot(X, weight) + bias - y) 
    w_=weight.copy()
    
    """
    for k in range(d):
        #calculate new w_[k]
        w_k= weight[k] - 2 * eta * np.sum(X[:, k] * (np.dot(X, weight) + bias - y))
        
        #update weights based on conditions
        if w_k < -2 * eta *_lambda:
            w_[k] = w_k+ 2 * eta * _lambda
        elif w_k > 2 * eta * _lambda:
            w_[k] = w_k- 2 * eta * _lambda
        else:
            w_[k]=0
    """
    #this is faster!
    #calculate w'
    gradient = np.dot(X.T, (np.dot(X, weight) + bias - y))
    w_ = weight - 2 * eta * gradient
    
    #updates w' same way as piecewise definition
    w_ = np.sign(w_) * np.maximum(np.abs(w_) - 2 * eta * _lambda, 0) #if |w'|-2*eta*lambda is less than 0, then w' is in the range where we set it =0
    
    return (w_, b_)

    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    return np.sum(((np.dot(X,weight) + bias) - y)**2) + _lambda * np.sum(np.abs(weight))
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001, #originally 0.00001
    convergence_delta: float = 1e-4, #originally 1e-4
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        weight = np.zeros(X.shape[1])
    else:
        weight = np.copy(start_weight)
    
    if start_bias is None:
        bias = 0
    else:
        bias = start_bias
    
    old_w = np.copy(weight)
    old_b = bias
    
    while True:
        weight, bias = step(X, y, weight, bias, _lambda, eta)
        
        if convergence_criterion(weight, old_w, bias, old_b, convergence_delta):
            break
        
        old_w = np.copy(weight)
        old_b = bias
    
    return weight, bias
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    weight_max = np.max(np.abs(weight-old_w))
    bias_max = np.max(np.abs(bias-old_b))
    
    if weight_max > convergence_delta or bias_max > convergence_delta: #if one or both have not converged
        return False
    else: #if both have converged we're good
        return True
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    #function to calculate FDR and TPR at each lambda step, using trained_weight and actual weight
    def calculate_f_t(trained_weight, weight, k):
        #number of incorrect zeros, trained is not zero but actual is zero
        #f = np.sum((trained_weight != 0) & weight ==0)
        f = np.sum(np.logical_and(trained_weight != 0, weight == 0))

        #number of correct NONzeros
        #t= np.sum((trained_weight != 0) & weight !=0)
        t = np.sum(np.logical_and(trained_weight != 0, weight != 0))
        
        if f + t > 0:
            fdr = f / (f + t) #divide by total # nonzeros
        else:
            fdr = 0
        
        if k > 0:
            tpr = t / k
        else:
            tpr = 0
        
        return fdr, tpr
    
    #create synthetic data
    n=500
    d=1000
    k=100
    
    weight=np.zeros(d) #k+1,...,d entries are zero
    weight[:k]= np.arange(1, k+1) #rest are weight[j]=j/k
    
    #both normal 0,1 random vars
    X = np.random.normal(0, 1, (n, d))
    epsilon = np.random.normal(0,1,n)
    
    #standardize X (maybe I don't have to do this if it's already std gaussian?)
    #X_std = (X-np.mean(X, axis=0))/np.std(X, axis=0)
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10) #for divide by zero errors
    
    #y_i is the data we wanted to generate using X and e
    y = np.dot(X,weight) + epsilon
    
    #start at lambda_max
    lambda_max= np.max(2 * np.abs(np.dot(X.T,y-np.mean(y)))) #equation(2)
    
    #collect nonzero count for various values of lambda
    lambdas = []
    nonzeros = []
    
    #keep track of fdr and tpr as we iterate over lambda
    fdr=[]
    tpr=[]
    
    _lambda= lambda_max
    
    while _lambda>0.01:
        lambdas.append(_lambda)
        
        #train with given lambda
        trained_weight, trained_bias = train(X_std, y, _lambda=_lambda)
        
        #count nonzero entries
        nonzeros.append(np.sum(trained_weight != 0))
        
        #collect fdr and tpr
        fdr_,tpr_=calculate_f_t(trained_weight, weight, k)
        
        fdr.append(fdr_)
        tpr.append(tpr_)
        
        #decrease lambda by factor of 2 as suggested
        _lambda /= 2
    
        print(_lambda)
        print(np.sum(trained_weight != 0))
        #print(trained_weight)
    
    #part A5(a), plot number of zero entries in weight as a function of lambda
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(lambdas, nonzeros)
    plt.xscale('log') #as suggested
    plt.xlabel('Lambda')
    plt.ylabel('number of nonzero entries in weight vector')
    plt.title('Number of nonzero weights vs. Lambda')
    
    #part A5(b), plot fdr vs tpr
    plt.subplot(1,2,2)
    plt.plot(fdr, tpr)
    plt.xlabel('False Discovery Rate (FDR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('FDR vs TPR')
    
    plt.show()

    #raise NotImplementedError("Your Code Goes Here")
    
if __name__ == "__main__":
    main()
