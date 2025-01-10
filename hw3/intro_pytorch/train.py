from typing import Dict, List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import problem


@problem.tag("hw3-A")
def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
) -> Dict[str, List[float]]:
    """Performs training of a provided model and provided dataset.

    Args:
        train_loader (DataLoader): DataLoader for training set.
        model (nn.Module): Model to train.
        criterion (nn.Module): Callable instance of loss function, that can be used to calculate loss for each batch.
        optimizer (optim.Optimizer): Optimizer used for updating parameters of the model.
        val_loader (Optional[DataLoader], optional): DataLoader for validation set.
            If defined, if should be used to calculate loss on validation set, after each epoch.
            Defaults to None.
        epochs (int, optional): Number of epochs (passes through dataset/dataloader) to train for.
            Defaults to 100.

    Returns:
        Dict[str, List[float]]: Dictionary with history of training.
            It should have have two keys: "train" and "val",
            each pointing to a list of floats representing loss at each epoch for corresponding dataset.
            If val_loader is undefined, "val" can point at an empty list.

    Note:
        - Calculating training loss might expensive if you do it seperately from training a model.
            Using a running loss approach is advised.
            In this case you will just use the loss that you called .backward() on add sum them up across batches.
            Then you can divide by length of train_loader, and you will have an average loss for each batch.
        - You will be iterating over multiple models in main function.
            Make sure the optimizer is defined for proper model.
        - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
            You might find some examples/tutorials useful.
            Also make sure to check out torch.no_grad function. It might be useful!
        - Make sure to load the model parameters corresponding to model with the best validation loss (if val_loader is provided).
            You might want to look into state_dict: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        
        from readme:    
    Finally you will use your implementations on a simple yet difficult problem.
    Given a dataset representing XOR function (we treat positives as truth, negatives as false), you will try total of 10 different architectures (5 for each loss function) and determine which one performs the best.

    To start look into `train` function in [train](./train.py) file.
    Here you will build a training loop through the dataset.
    At the end of each epoch, you will record running training loss, and validation loss, if validation loader has been provided.
    """
    #need to output train and val
    output = {"train": [], "val": []}
    
    #record best model/best validation loss (at any part during training) for part c
    best_model = None
    best_validation_loss = float('inf')
    
    #track best model based on training loss
    for epoch in range(epochs):
        model.train() #train mode
        total_loss = 0 #track loss over training iterations below
        
        for x,y in train_loader:
            optimizer.zero_grad() #make sure gradients reset after each step
        
            #forward
            loss = criterion(model(x), y) #criterion is loss function
            
            #backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        #find epoch avg training loss
        epoch_t_loss = total_loss/len(train_loader)
        output["train"].append(epoch_t_loss)

    #track best model based on validation loss (if val_loader is passed in)
        if val_loader is not None:
            model.eval() #evaluation mode
            v_loss = 0
            
            #no_grad saves time https://pytorch.org/docs/stable/generated/torch.no_grad.html#no-grad
            with torch.no_grad():
                #forward (no backward step for validation)
                for x,y in val_loader:
                    loss= criterion(model(x),y)
                    v_loss += loss.item()
            
            epoch_v_loss = v_loss/len(val_loader)
            output["val"].append(epoch_v_loss)
        
            #is this new validation loss better than the previous best?
            if epoch_v_loss < best_validation_loss:
                best_validation_loss=epoch_v_loss
                best_model = model.state_dict()
                
                #make sure that the best model is the current state at the end of training
    if val_loader is not None and best_model is not None:
        model.load_state_dict(best_model)
        
    return output
    #raise NotImplementedError("Your Code Goes Here")


def plot_model_guesses(
    dataloader: DataLoader, model: nn.Module, title: Optional[str] = None
):
    """Helper function!
    Given data and model plots model predictions, and groups them into:
        - True positives
        - False positives
        - True negatives
        - False negatives

    Args:
        dataloader (DataLoader): Data to plot.
        model (nn.Module): Model to make predictions.
        title (Optional[str], optional): Optional title of the plot.
            Might be useful for distinguishing between MSE and CrossEntropy.
            Defaults to None.
    """
    with torch.no_grad():
        list_xs = []
        list_ys_pred = []
        list_ys_batch = []
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            list_xs.extend(x_batch.numpy())
            list_ys_batch.extend(y_batch.numpy())
            list_ys_pred.extend(torch.argmax(y_pred, dim=1).numpy())

        xs = np.array(list_xs)
        ys_pred = np.array(list_ys_pred)
        ys_batch = np.array(list_ys_batch)

        # True positive
        if len(ys_batch.shape) == 2 and ys_batch.shape[1] == 2:
            # MSE fix
            ys_batch = np.argmax(ys_batch, axis=1)
        idxs = np.logical_and(ys_batch, ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="o", c="green", label="True Positive"
        )
        # False positive
        idxs = np.logical_and(1 - ys_batch, ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="o", c="red", label="False Positive"
        )
        # True negative
        idxs = np.logical_and(1 - ys_batch, 1 - ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="x", c="green", label="True Negative"
        )
        # False negative
        idxs = np.logical_and(ys_batch, 1 - ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="x", c="red", label="False Negative"
        )

        if title:
            plt.title(title)
        plt.xlabel("x0")
        plt.ylabel("x1")
        plt.legend()
        plt.show()
