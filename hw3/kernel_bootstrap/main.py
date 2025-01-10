from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 6 * np.sin(np.pi * x) * np.cos(4 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (1+np.multiply.outer(x_i,x_j))**d
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp((-gamma* (np.subtract.outer(x_i,x_j)**2)))
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
        
        note:
        alpha hat is similar to the ridge regression: a=(K+lambda*I)^{-1}y
    """
    n = len(x) #also len(y)
    
    #calculate kernel
    K= kernel_function(x,x,kernel_param) #uses whichever kernel function is specified
    
    return np.dot(np.linalg.inv((K+_lambda * np.eye(n,n))),y)
    
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across all folds.
    """
    fold_size = len(x) // num_folds
    
    #keep track of loss
    loss = []
    
    for i in range(num_folds):
        #increment over fold_size sized subsets
        start = i * fold_size #beginning of fold
        end = (i+1) * fold_size #end of fold
        
        #validation sets
        x_fold= x[start:end]
        y_fold=y[start:end]
        
        #delete folds/validation sets to keep just training data
        x_train = np.delete(x, np.array(range(start,end,1)), 0)
        y_train = np.delete(y, np.array(range(start,end,1)), 0)
        
        #train/calculate alpha hat
        alpha_hat = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        
        #calculate predicted y based on alpha
        y_pred = np.dot(alpha_hat, kernel_function(x_train, x_fold, kernel_param))
        
        #calculate error on y for validation set
        loss_i= np.mean((y_fold-y_pred)**2)
        loss.append(loss_i)
        
    return np.mean(loss)        
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    #will do random search
    
    #start at median gamma
    dist = np.subtract.outer(x, x) ** 2
    gamma = 1 / np.median(dist[np.triu_indices(dist.shape[0])]) #heuristic from footnote
    
    best_lambda = None
    best_loss = float('inf')

    num_iterations = 1000 
    
    for _ in range(num_iterations):
        #sampling lambda from distribution 10**i, where i~Unif(-5, -1), as specified
        _lambda = 10 ** np.random.uniform(-5, -1)
        
        #will not search over gamma
        
        #cross-validation with current params
        loss = cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)
        
        #update lambda if loss is better than previous best
        if loss < best_loss:
            best_loss = loss
            best_lambda = _lambda
    
    return best_lambda, gamma

    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    #will do random search
    
    best_lambda = None
    best_d = None
    best_loss = float('inf')

    num_iterations = 1000 
    
    for _ in range(num_iterations):
        #sampling lambda from distribution 10**i, where i~Unif(-5, -1), as specified
        _lambda = 10 ** np.random.uniform(-5, -1)
        
        #sampling d from distribution [5, 6, ..., 24, 25]
        d=np.random.choice(np.arange(5,25))
        
        #cross-validation with current params
        loss = cross_validation(x, y, poly_kernel, d, _lambda, num_folds)
        
        #update lambda if loss is better than previous best
        if loss < best_loss:
            best_loss = loss
            best_lambda = _lambda
            best_d = d
    
    return best_lambda, best_d
    #raise NotImplementedError("Your Code Goes Here")

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap") #but only use x_30 and y_30
    
    # part a finding best params for each 
    lambda_poly, d_poly = poly_param_search(x_30, y_30, num_folds=len(x_30))
    lambda_rbf, gamma_rbf = rbf_param_search(x_30, y_30, num_folds=len(x_30))
    
    print('polynomial parameters:')
    print(lambda_poly, d_poly)
    
    print('rbf parameters:')
    print(lambda_rbf, gamma_rbf)
    
    #part b plot original data, f, and f^hat
    #fine grid can be defined as np.linspace(0, 1, num=100)
    fine_grid = np.linspace(0, 1, num=100)
    
    #predict poly
    poly_alpha = train(x_30, y_30, poly_kernel, d_poly, lambda_poly)
    pred_poly = np.dot(poly_kernel(fine_grid, x_30, d_poly), poly_alpha)
    
    #predict rbf
    rbf_alpha = train(x_30, y_30, rbf_kernel, gamma_rbf, lambda_rbf)
    pred_rbf = np.dot(rbf_kernel(fine_grid, x_30, gamma_rbf), rbf_alpha)
    
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 2)
    plt.scatter(x_30, y_30, label='original data')
    plt.plot(fine_grid, f_true(fine_grid), label='true f')
    plt.plot(fine_grid, pred_poly, label='predicted f')
    plt.title("Polynomial Kernel Plot")
    plt.ylim(-6, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()  
    
    plt.subplot(1, 2, 1)
    plt.scatter(x_30, y_30, label='original data')
    plt.plot(fine_grid, f_true(fine_grid), label='true f')
    plt.plot(fine_grid, pred_rbf, label='predicted f')
    plt.title("RBF Kernel Plot")
    plt.ylim(-6, 6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()  

    plt.show()
    
    #raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
