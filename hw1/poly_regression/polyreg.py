"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields (I did not add additional fields)
        #raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).a
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        #return vandermonde matrix, minus the first column (I'm a math 318 TA hehe)
        #nevermind this made me fail the tests :/
        #X_flat=X.flatten()
        #return np.vander(X_flat, degree, increasing=True)[:, 1:]
        
        n = len(X)
        X_flat = X.flatten()
        
        result = np.zeros((n, degree))
        for i in range(1, degree + 1):
            result[:, i-1] = X_flat ** i
        return result
        

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        #copying some parts from linreg_closedform.py
        n = len(X)      
          
        #polynomial expansion
        X_expand = self.polyfeatures(X, self.degree)
        
        #standardize before adding bias term (standardization procedure per Ed post)
        self.mean = np.mean(X_expand, axis=0)
        self.std = np.std(X_expand, axis=0)
        X_expand = (X_expand- self.mean)/self.std
        
        # add 1s column (after standardizing)
        X_expand = np.c_[np.ones([n, 1]), X_expand]
        
        n, d = X_expand.shape
        #idk why we subtract 1 from d to add 1 back later so I'll just not
        
        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        #self.theta = np.linalg.solve(X_expand.T @ X_expand + reg_matrix, X_expand.T @ y)
        self.weight = np.linalg.pinv(X_expand.T @ X_expand + reg_matrix) @ X_expand.T @ y



    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        n = len(X)      
          
        #polynomial expansion
        X_expand = self.polyfeatures(X, self.degree)
        
        #standardizing
        X_expand = (X_expand- self.mean)/self.std
        
        # add 1s column (after standardizing)
        X_expand = np.c_[np.ones([n, 1]), X_expand]
        
        #multiply the scaled vandermonde-ish matrix by the weights
        return X_expand @ self.weight


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    #find error
    diff = a-b
    
    #calculate mean square error
    return np.mean(np.square(diff))


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    
    #initialize polynomial regression model
    model = PolynomialRegression(degree, reg_lambda)
    
    for i in range(1,n):
        #fitting the model
        #start at i+1 since "errorTrain[0:1] and errorTest[0:1] won't actually matter"
        model.fit(Xtrain[0:i+1],Ytrain[0:i+1])
        
        #use model to generate predictions
        train_i = model.predict(Xtrain[0:i+1])
        test_i = model.predict(Xtest)
        
        #calculate mean squared error
        #ith index corresponds to i+1th error
        errorTrain[i] = mean_squared_error(train_i, Ytrain[0:i+1])
        errorTest[i] = mean_squared_error(test_i, Ytest)
    
   # errorTrain=mean_squared_error(predict(self,X), Xtrain)
    
    return errorTrain, errorTest
