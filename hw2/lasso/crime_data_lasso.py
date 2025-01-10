if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    
    #train data
    y = df_train['ViolentCrimesPerPop'] #ViolentCrimesPerPop is the response variable
    X = df_train.drop('ViolentCrimesPerPop', axis=1)
    
    #test data
    y_test = df_test['ViolentCrimesPerPop'] #ViolentCrimesPerPop is the response variable
    X_test = df_test.drop('ViolentCrimesPerPop', axis=1)
    
    #start at lambda_max
    lambda_max= np.max(2 * np.abs(np.dot(X.T,y-np.mean(y)))) #equation(2)
    
    #collect nonzero count for various values of lambda
    lambdas = []
    nonzeros = []
    
    #collect regularization paths
    vars = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    var_indices = [X.columns.get_loc(v) for v in vars]
    regularization_path = {v: [] for v in vars}   
    
    #collect MSE for train and test set
    train_mse = []
    test_mse =[]
    
    _lambda= lambda_max
    
    while _lambda>0.01:
        lambdas.append(_lambda)
        
        #train with given lambda
        trained_weight, trained_bias = train(X, y, _lambda=_lambda)
        
        #count nonzero entries(c)
        nonzeros.append(np.sum(trained_weight != 0))
        
        #store regularization paths (d)
        for i, v in enumerate(vars):
            regularization_path[v].append(trained_weight[var_indices[i]])
            
        #calculate and store train and test MSE
        train_mse.append(np.mean((y-(X.dot(trained_weight) + trained_bias))**2))
        test_mse.append(np.mean((y_test-(X_test.dot(trained_weight) + trained_bias))**2))
        
        #decrease lambda by factor of 2 as suggested
        _lambda /= 2
    
        #sanity check to make sure my code is actually running :/ 
        print(_lambda)
        print(np.sum(trained_weight != 0))
        #print(trained_weight)
    
    #part A6(c), plot number of zero entries in weight as a function of lambda
    plt.figure(figsize=(18,10))
    plt.subplot(1,2,1)
    plt.plot(lambdas, nonzeros)
    plt.xscale('log') #as suggested
    plt.xlabel('Lambda')
    plt.ylabel('number of nonzero entries in weight vector')
    plt.title('Lambda vs nonzero weights for crime data')
    
    #part A6(d), plot regularization paths for certain vars (as a function of lambda)
    plt.subplot(1,2,2)
    for v in vars:
        plt.plot(lambdas, regularization_path[v], label=v)
    plt.xscale('log') #as suggested
    plt.xlabel('Lambda')
    plt.ylabel('Weight')
    plt.title('Regularization paths')
    plt.legend()
    
    #part A6(e), plot the squared error on the training and test data as a function of λ
    plt.subplot(2,2,1)
    plt.plot(lambdas, train_mse, label="training")
    plt.plot(lambdas, test_mse, label="testing")
    plt.xscale('log') #as suggested
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.title('Train and test MSE for varying Lambda')
    plt.legend()
    
    #plt.show()
    
    #part A6(f) retrain with \lambda = 30
    w,b=train(X, y, _lambda=30)
    for label, weight in zip(X.columns, w):
        print(f"{label}: {weight}")
    print(f"\nMost Positive Weight: {X.columns[np.argmax(w)]}: {w[np.argmax(w)]}")
    print(f"Most Negative Weight: {X.columns[np.argmin(w)]}: {w[np.argmin(w)]}")

    #raise NotImplementedError("Your Code Goes Here")

if __name__ == "__main__":
    main()
