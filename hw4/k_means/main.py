if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. Make sure to change it back before submission!
    """
    #load data
    (x_train, _), _ = load_dataset("mnist")
    
    #for debugging
    #x_train = x_train[:10000]
    
    #need to reshape image data into a row
    x_train_row = x_train.reshape(x_train.shape[0], -1)
    
    #10 centers
    k=10
    
    #run algorithm
    centers, error = lloyd_algorithm(x_train_row, k)
    
    #reshape images (of centers, as specified) back to square
    images = centers.reshape(k, 28,28)
    
    #plot images of each center
    plt.figure(figsize=(20, 2))
    for i in range(k):
        plt.subplot(1, k, i + 1)
        plt.imshow(images[i])
    plt.show()
    
    #raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
