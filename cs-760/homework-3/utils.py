import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

def sigmoid(model, x):
    """Compute the sigmoid function."""
    #Return the sigmoid function
    return 1 / (1 + np.exp(-np.dot(model, x)))

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
    y_grid = KNeighborsClassifier(n_neighbors=k).fit(train_data[['x1','x2']],train_data['y']).predict(X_grid)
    plt.scatter(x1, x2, c=y_grid, cmap=plt.cm.Paired)
    # Show training data
    plt.scatter(train_data['x1'], train_data['x2'], c=train_data['y'], cmap=plt.cm.Paired, edgecolors='black')
    plt.show()

def knn_cross_val(k, data, y_column, num_folds=5, output_file=None):
    """Perform k-NN cross-validation."""
    #Return the average cross-validation error
    #Split the data into num_folds folds
    folds = np.array_split(data, num_folds)
    classifier = KNeighborsClassifier(n_neighbors=k)
    #For each fold, train on the other folds and test on the current fold
    
    # Get accuracy, precision and recall for each fold
    accuracy_list = []
    precision_list = []
    recall_list = []

    for i in range(num_folds):
        train = pd.concat([folds[j] for j in range(num_folds) if j != i])
        test = folds[i]
        #Compute the error for each fold
        y_pred = classifier.fit(train.drop(y_column, axis=1),train[y_column]).predict(test.drop(y_column, axis=1))
        #Calculate TPR FPR TNR FNR
        tp = np.sum(y_pred[y_pred == 1] == test[y_column][y_pred == 1])
        fp = np.sum(y_pred[y_pred == 1] != test[y_column][y_pred == 1])
        tn = np.sum(y_pred[y_pred == 0] == test[y_column][y_pred == 0])
        fn = np.sum(y_pred[y_pred == 0] != test[y_column][y_pred == 0])

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        print("Fold {}: accuracy = {}, precision = {}, recall = {}".format(i+1, accuracy, precision, recall))

    if output_file:
        df = pd.DataFrame({'accuracy': accuracy_list, 'precision': precision_list, 'recall': recall_list})
        df['fold'] = np.arange(1, num_folds+1)
        df.to_csv(output_file, index=False)
    # Return the average accuracy, precision and recall
    return np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list)


def logistic_regression_cross_val(data, eta, y_column, num_folds=5, epochs=5000, output_file=None):
    """Perform logistic regression cross-validation."""
    # Initialize model with 0 weights with the same number of features as the data
    model = np.zeros(data.shape[1]-1)
    folds = np.array_split(data, num_folds)

    # Get accuracy, precision and recall for each fold
    accuracy_list = []
    precision_list = []
    recall_list = []
    print("Logistic regression with eta = {}, {} folds".format(eta, num_folds))
    for i in range(num_folds):
        train = pd.concat([folds[j] for j in range(num_folds) if j != i])
        train_features = train.drop(y_column, axis=1)
        train_y = train[y_column]
        
        test = folds[i]
        test_features = test.drop(y_column, axis=1)
        test_y = test[y_column]
        for j in range(epochs):
            # Take a random training example (SGD)
            index = np.random.randint(0, train.shape[0])
            # Compute gradient
            gradient = (sigmoid(model, train_features.iloc[index])-train_y.iloc[index]) * train_features.iloc[index]
            # Update model
            model -= eta * gradient
        # Compute accuracy, precision and recall
        y_pred = test_features.apply(lambda x: sigmoid(model, x), axis=1)
        #Calculate TPR FPR TNR FNR
        tp = np.sum(y_pred[y_pred == 1] == test[y_column][y_pred == 1])
        fp = np.sum(y_pred[y_pred == 1] != test[y_column][y_pred == 1])
        tn = np.sum(y_pred[y_pred == 0] == test[y_column][y_pred == 0])
        fn = np.sum(y_pred[y_pred == 0] != test[y_column][y_pred == 0])
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        print("Fold {}: accuracy = {}, precision = {}, recall = {}".format(i+1, accuracy, precision, recall))
    # Return the average accuracy, precision and recall
    return np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list)

    

def roc_curves(k, train_data, test_data, y_column, eta = 0.01, epochs=5000, output_file = None):
    # Plot ROC curves for kNN and logistic regression

    # kNN
    y_pred = KNeighborsClassifier(n_neighbors=k).fit(train_data.drop(y_column, axis=1),train_data[y_column]).predict(test_data.drop(y_column, axis=1))
    fpr, tpr, thresholds = roc_curve(test_data[y_column], y_pred)
    plt.plot(fpr, tpr, label='kNN')

    # Logistic regression
    train_y = train_data[y_column]
    train_features = train_data.drop(y_column, axis=1)
    model = np.zeros(train_data.shape[1]-1)
    for j in range(epochs):
        # Take a random training example
        index = np.random.randint(0, train_data.shape[0])
        # Compute gradient
        gradient = (sigmoid(model, train_features.iloc[index])-train_y.iloc[index]) * train_features.iloc[index]
        # Update model
        model -= eta * gradient

    y_pred = test_data.drop(y_column, axis=1).apply(lambda x: sigmoid(model, x), axis=1)
    fpr, tpr, thresholds = roc_curve(test_data[y_column], y_pred)
    plt.plot(fpr, tpr, label='Logistic regression')
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curves')
    plt.show()
    
