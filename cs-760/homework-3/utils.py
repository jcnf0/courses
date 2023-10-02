import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt

def get_data(file):
    """Get data from file."""
    #Return x1, x2 and y in a dataframe
    data = np.loadtxt(file)
    df = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
    return df

def plot_knn_predictions(grid_limits,grid_step,k,train_data):
    """Plot the predictions of k-NN on a grid."""
    #Return a plot of the predictions of k-NN on a grid
    x1_min, x1_max, x2_min, x2_max = grid_limits
    x1_step, x2_step = grid_step
    x1 = np.arange(x1_min, x1_max, x1_step)
    x2 = np.arange(x2_min, x2_max, x2_step)
    x1, x2 = np.meshgrid(x1, x2)
    x1 = x1.ravel()
    x2 = x2.ravel()
    X_grid = np.array([x1, x2]).T
    y_grid = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k).fit(train_data[['x1','x2']],train_data['y']).predict(X_grid)
    plt.scatter(x1, x2, c=y_grid, cmap=plt.cm.Paired)
    plt.show()