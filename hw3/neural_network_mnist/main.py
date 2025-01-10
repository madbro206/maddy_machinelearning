# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer (hm i should have read that earlier)
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension. 64
            d (int): Input dimension/number of features. 784 input
            k (int): Output dimension/number of classes. 10 output
        """
        super().__init__()
        
        #layer 0
        alpha_0 = 1/(d ** 0.5)
        self.W_0 = Parameter(Uniform(-alpha_0, alpha_0).sample((d,h))) #initialize according to Unif(-alpha, alpha)
        self.b_0 = Parameter(Uniform(-alpha_0, alpha_0).sample((h,))) #h is output (to hidden layer)
        
        #layer 1
        alpha_1 = 1/(h ** 0.5) #h is input to hidden layer (makes sense)
        self.W_1 = Parameter(Uniform(-alpha_1, alpha_1).sample((h,k))) #initialize according to Unif(-alpha, alpha)
        self.b_1 = Parameter(Uniform(-alpha_1, alpha_1).sample((k,)))
        
        #raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        layer_0 = x @ self.W_0 + self.b_0
        
        return relu(layer_0) @ self.W_1 + self.b_1 #use relu for non-linearity 
        
        #raise NotImplementedError("Your Code Goes Here")


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer). 32
            h1 (int): Second hidden dimension (between second and third layer). 32
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        
        #layer 0
        alpha_0 = 1/(d ** 0.5)
        self.W_0 = Parameter(Uniform(-alpha_0, alpha_0).sample((d,h0))) #initialize according to Unif(-alpha, alpha)
        self.b_0 = Parameter(Uniform(-alpha_0, alpha_0).sample((h0,))) #h is output (to hidden layer)
        
        #layer 1
        alpha_1 = 1/(h0 ** 0.5)
        self.W_1 = Parameter(Uniform(-alpha_1, alpha_1).sample((h0,h1))) #initialize according to Unif(-alpha, alpha)
        self.b_1 = Parameter(Uniform(-alpha_1, alpha_1).sample((h1,))) #h is output (to hidden layer)
        
        #layer 2
        alpha_2 = 1/(h1 ** 0.5)
        self.W_2 = Parameter(Uniform(-alpha_2, alpha_2).sample((h1,k))) #initialize according to Unif(-alpha, alpha)
        self.b_2 = Parameter(Uniform(-alpha_2, alpha_2).sample((k,))) #h is output (to hidden layer)
        
        #raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        layer_0 = x @ self.W_0 + self.b_0
        
        layer_1 = relu(layer_0) @ self.W_1 + self.b_1
        
        return relu(layer_1) @ self.W_2 + self.b_2 #use relu for non-linearity         
        #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    epochs = 100 #?
    loss = [] #record avg loss for each epoch
    
    for epoch in range(epochs):
        total_loss = 0 #total loss (for this epoch)
        correct_pred = 0
        samples = 0
        
        #iterate over data (train_loader)
        for x, y in train_loader:
            optimizer.zero_grad() #make sure gradients reset after each step
            
            #forward
            y_pred = model(x)
            current_loss = cross_entropy(y_pred, y) #use crossentropy losses
            
            #backward
            current_loss.backward()
            optimizer.step()
            
            total_loss += current_loss.item() * x.size(0) #loss per sample
            
            #track correct predictions for accuracy calc
            correct_pred += torch.sum(torch.argmax(y_pred, dim=1)==y)
            samples += y.size(0)

        #calculate avergage loss and overall accuracy for this epoch
        avg_loss = total_loss/samples
        accuracy = correct_pred/samples
    
        loss.append(avg_loss)
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {accuracy}")
        
        #stop when accuracy is greater than 99%
        if accuracy > 0.99:
            break
    
    return loss
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
#part a, F1 model
    f1_model = F1(64, 784, 10) #dimensions as specified
    f1_optimizer = Adam(f1_model.parameters(),lr=0.001) #change learning rate? i am supposed to choose it
    
    train_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)
    
    f1_loss = train(f1_model, f1_optimizer, train_loader)
    
    # plot loss vs epoch after 99% accuracy is reached
    plt.plot(f1_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("F! Training Loss per Epoch")
    plt.show()
    
    #"Finally evaluate the model on the test data and report both the accuracy and the loss."
    correct_predictions = torch.sum(torch.argmax(f1_model(x_test), dim=1) == y_test)
    test_accuracy = correct_predictions.item() / len(x_test)

    test_loss = cross_entropy(f1_model(x_test), y_test)
    
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Loss: {test_loss}")
    
#part b, F2 model   
    f2_model = F2(32,32,784,10) #dimensions as specified
    f2_optimizer = Adam(f2_model.parameters(),lr=0.001) 
    
    f2_loss = train(f2_model, f2_optimizer, train_loader)
    
    # plot loss vs epoch after 99% accuracy is reached
    plt.plot(f2_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("F2 Training Loss per Epoch")
    plt.show()
    
    #"Finally evaluate the model on the test data and report both the accuracy and the loss."
    correct_predictions2 = torch.sum(torch.argmax(f2_model(x_test), dim=1) == y_test)
    test_accuracy2 = correct_predictions.item() / len(x_test)

    test_loss2 = cross_entropy(f2_model(x_test), y_test)
    
    print(f"Test Accuracy (F2): {test_accuracy2}")
    print(f"Test Loss (F2): {test_loss2}")
    
    
#part c, counting parameters
    f1_params = sum(p.numel() for p in f1_model.parameters())
    f2_params = sum(p.numel() for p in f2_model.parameters())
    
    print(f"F1 parameters: {f1_params}")
    print(f"F2 parameters: {f2_params}")
    #raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
