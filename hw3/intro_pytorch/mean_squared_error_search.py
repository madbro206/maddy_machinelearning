if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    model.eval() #evaluation mode
    
    #count corect and total predictions
    corr_pred = 0
    total_pred = 0
    
    for x,y in dataloader:
        predictions = (model(x) > 0.5).float() #different target than crossentropy
        
        corr_pred += (predictions == y).all(dim=1).sum().item()
        total_pred += y.size(0)
    
    accuracy = corr_pred/total_pred
    return accuracy
    
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    #will return history of training of all models
    history = {}
    criterion = MSELossLayer()
    
    #dataloaders (different for crossentropy vs MSE)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True) #change batch size?
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
    
    #1 Linear neural network (Single layer, no activation function)
    #use Sequential to compose modules together, per section 6
    model_1 = nn.Sequential(
        LinearLayer(2,2) #binary classification, two inputs and two outputs
    ) #similar to crossentropy but no softmax layer
 
    optimizer = SGDOptimizer(model_1.parameters(), lr=0.01) #change learning rate?
    
    #call train with the correct optimizer and stuff, add to history
    current_hist = train(train_loader, model_1, criterion, optimizer, val_loader, epochs=100) #epochs=100
    
    history["linear_model"] = {"train": current_hist["train"], "val": current_hist["val"], "model": model_1}   
    
    #2 NN with one hidden layer (2 units) and sigmoid activation function after the hidden layer
    model_2 = nn.Sequential(
        LinearLayer(2,2),
        SigmoidLayer(),
        LinearLayer(2,2)
    )
    
    optimizer = SGDOptimizer(model_2.parameters(), lr=0.01) #change learning rate?
    current_hist = train(train_loader, model_2, criterion, optimizer, val_loader, epochs=100) #epochs=100 
    history["sigmoid_model"] = {"train": current_hist["train"], "val": current_hist["val"], "model": model_2}   
    
    #3 NN with one hidden layer (2 units) and ReLU activation function after the hidden layer
    model_3 = nn.Sequential(
            LinearLayer(2,2),
            ReLULayer(),
            LinearLayer(2,2)
    )
    
    optimizer = SGDOptimizer(model_3.parameters(), lr=0.01) #change learning rate?
    current_hist = train(train_loader, model_3, criterion, optimizer, val_loader, epochs=100) #epochs=100 
    history["relu_model"] = {"train": current_hist["train"], "val": current_hist["val"], "model": model_3}     
    
    #4 NN with two hidden layer (each with 2 units) and Sigmoid, ReLU activation functions after first and second hidden layers, respectively
    model_4 = nn.Sequential(
        LinearLayer(2,2),
        SigmoidLayer(),
        LinearLayer(2,2),
        ReLULayer(),
        LinearLayer(2,2)
    )
    
    optimizer = SGDOptimizer(model_4.parameters(), lr=0.01) #change learning rate?
    current_hist = train(train_loader, model_4, criterion, optimizer, val_loader, epochs=100) #epochs=100 
    history["sig_relu_model"] = {"train": current_hist["train"], "val": current_hist["val"], "model": model_4} 
    
    #5 NN with two hidden layer (each with 2 units) and ReLU, Sigmoid activation functions after first and second hidden layers, respectively
    model_5 = nn.Sequential(
        LinearLayer(2,2),
        ReLULayer(),
        LinearLayer(2,2),
        SigmoidLayer(),
        LinearLayer(2,2)
    )
    
    optimizer = SGDOptimizer(model_5.parameters(), lr=0.01) #change learning rate?
    current_hist = train(train_loader, model_5, criterion, optimizer, val_loader, epochs=100) #epochs=100 
    history["relu_sig_model"] = {"train": current_hist["train"], "val": current_hist["val"], "model": model_5}    
    
    return history
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )
    
 ### part b ###
    #1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
    mse_configs = mse_parameter_search(dataset_train, dataset_val)
    
    # 2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            #x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
    plt.figure(figsize=(6, 6))
    for model, data in mse_configs.items():
        plt.plot(data['train'], label=f'{model} (train)')
        plt.plot(data['val'], label=f'{model} (val)')

    plt.title('Train and Validation losses for MSE models')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.show()   
    
 ### part c ###
    
    #3. Choose and report the best model configuration based on validation losses.
           # In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
    best_model =None
    best_loss=float('inf')
    best_model_actual = None
           
    for model, data in mse_configs.items():
        min_val = min(data['val']) 
            
        #update best model when looking through all models and losses    
        if best_loss > min_val:
            best_model = model #save model name
            best_model_actual = data['model'] #save best model
            
            best_loss = min_val
            
    print("Best model and validation loss from MSE:")
    print(best_model)
    print(best_loss)
           
    #4. Plot best model guesses on test set (using plot_model_guesses function from train file)
    best_model_actual.eval()
    
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    plot_model_guesses(test_loader, best_model_actual)
    
    #5. Report accuracy of the model on test set.
    print("Test accuracy of best model from MSE:")
    print(accuracy_score(best_model_actual, test_loader))   
    
    
    #raise NotImplementedError("Your Code Goes Here")


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
