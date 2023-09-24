import numpy as np
from scipy.interpolate import lagrange
import pandas as pd
import matplotlib.pyplot as plt
import os 

class Node():
    def __init__(self, y, feature=None, threshold=None, left=None, right=None):
        self.value = self._most_common_label(y)
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = self.left is None and self.right is None
    
    # Find most common label
    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)

        # We predict y=1 if there is no majority class in the leaf.
        if len(counts) == 2 and counts[0]==counts[-1]:
            return 1
        return unique[np.argmax(counts)]

class DecisionTree():
    def __init__(self):
        self.root = None
        self.num_nodes = 0
        
    def fit(self, data):
        self.root = self._build_tree(data[["x1","x2"]], data["y"])

    def predict(self, X):
        return X.apply(lambda x: self._predict(x, self.root), axis=1)
    
    # Calculate error (MSE)
    def err_n(self, test_data):
        y_pred = self.predict(test_data[["x1","x2"]])
        y = test_data["y"]
        return ((y_pred - y)**2).mean()
    
    def plot_tree(self, output_file=None):
        tree = self._plot_tree(self.root)
        print(tree)
        if output_file and tree is not None:
            # Check if directory exists
            directory = os.path.dirname(output_file)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(output_file, "w") as f:
                f.write(tree)
        return 

    # Build using information gain
    def _build_tree(self, X, y):
        best_feature, best_threshold = self._choose_split(X, y)
        if best_feature is None or best_threshold is None:
            return Node(y)
        left_indices = X[best_feature] < best_threshold
        right_indices = ~left_indices
        left = self._build_tree(X[left_indices], y[left_indices])
        right = self._build_tree(X[right_indices], y[right_indices])
        self.num_nodes += 1
        return Node(y, best_feature, best_threshold, left, right)
    
    # Choose split with highest information gain
    def _choose_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, 0.0
        for feature in ["x1", "x2"]:
            thresholds = np.unique(X[feature])
            for threshold in thresholds:
                gain = self._information_gain(y, threshold, X[feature])
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain

        if best_gain == 0:
            return None, None
        return best_feature, best_threshold
    
    # Calculate information gain
    def _information_gain(self, y, threshold, feature):
        parent_entropy = self._entropy(y)
        left_indices = feature < threshold
        right_indices = ~left_indices
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        child_entropy = (len(y[left_indices]) / len(y)) * left_entropy + (len(y[right_indices]) / len(y)) * right_entropy
        return parent_entropy - child_entropy
    
    # Calculate entropy
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy
    
    # Predict using tree
    def _predict(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)
    
    # Plot tree as graph with feature and threshold at each node
    def _plot_tree(self, node, depth=0):
        if node is None:
            return

        prefix = "|   " * depth
        if depth > 0:
            prefix += "|--- "

        if node.is_leaf:
            print(prefix + f"class: {int(node.value)}")
        else:
            feature_name = node.feature
            threshold = node.threshold
            print(f"{prefix}{feature_name} <= {threshold:.2f}")
            self._plot_tree(node.left, depth + 1)
            print(prefix + "|")
            print(f"{prefix}{feature_name} > {threshold:.2f}")
            self._plot_tree(node.right, depth + 1)

    
def get_data(file):
    """Get data from file."""
    #Return x1, x2 and y in a dataframe
    data = np.loadtxt(file)
    df = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
    return df

def data_split(file, split=8192, shuffle=True):
    """Split data into training and testing sets."""
    data = get_data(file)
    data.sample(frac=1).reset_index(drop=True)
    return data[:split], data[split:]

# Show data in scatterplot with different colors for each class
def show_data(data,output_file=None):
    plt.scatter(data["x1"], data["x2"], c=data["y"], cmap=plt.cm.Spectral, s=1)
    if output_file:
        # Check if directory exists
        directory = os.path.dirname(output_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(output_file)
        plt.clf()
    else:
        plt.show()
    return

def nested_sets(file, sizes=[32,128,512,2048,8192]):
    """Split data into training and testing sets."""
    data = get_data(file)
    sets = []
    data = data.sample(frac=1).reset_index(drop=True)
    for size in sizes:
        if size > len(data):
            raise ValueError("Size of set is larger than data.")
        sets.append(data[:size])
    test_set = data[len(data)-sizes[-1]:]
    return sets,test_set

def plot_decision_boundary(classifier, data, output_file=None,limits = None):
    """Plot the decision boundary of a classifier."""
    # Set min and max values and give it some padding
    x1_min, x1_max = data["x1"].min() - 1, data["x1"].max() + 1
    x2_min, x2_max = data["x2"].min() - 1, data["x2"].max() + 1

    h = 0.01

    # Generate a grid of points with distance h between them
    x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    # Create data frame with all points in grid
    grid = pd.DataFrame(np.c_[x1x1.ravel(), x2x2.ravel()], columns=["x1", "x2"])

    # Predict the function value for the whole grid
    Z=np.array(classifier.predict(grid)).reshape(x1x1.shape)

    # Plot the contour and training examples
    plt.contourf(x1x1, x2x2, Z, cmap=plt.cm.Spectral)
    plt.scatter(data["x1"], data["x2"], c=data["y"], cmap=plt.cm.Spectral, s=1)
    if limits:
        plt.xlim(limits[0], limits[1])
        plt.ylim(limits[2], limits[3])
    if output_file:
        # Check if directory exists
        directory = os.path.dirname(output_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(output_file)
        plt.clf()
    else:
        plt.show()
    
    return

def plot_error_n(classifiers, test_data, label = None,output_file=None):
    """Plot err_n vs n for a list of classifiers."""
    num_nodes_list=[]
    errn_list=[]

    for k in range(len(classifiers)):
        if label == "sklearn":
            num_nodes_list.append(classifiers[k].tree_.node_count)
            errn_list.append(1-classifiers[k].score(test_data[['x1','x2']],test_data['y']))
        else:
            num_nodes_list.append(classifiers[k].num_nodes)
            errn_list.append(classifiers[k].err_n(test_data))
    
    plt.scatter(num_nodes_list, errn_list, label=label)
    plt.xlabel('n')
    plt.ylabel('err_n')
    plt.legend()
    if output_file:
        # Check if directory exists
        directory = os.path.dirname(output_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(output_file)
        plt.clf()
    else:
        plt.show()
    return num_nodes_list, errn_list

def sample_uniform(a,b,n,mean=None,std=1):
    # Sample n points from [a,b] with noise N(mean,std)
    x = np.random.uniform(a,b,n)
    if mean is not None:
        x += np.random.normal(mean,std,n)
    
    y = np.sin(x)
    df = pd.DataFrame(np.c_[x,y], columns=['x', 'y'])
    return df

def lagrange_test(train,test):
    # Train a polynomial of degree 2 on train data
    model = lagrange(train['x'],train['y'])
    # Calculate MSE on test data
    return ((model(train['x'])-train['y'])**2).mean(), ((model(test['x'])-test['y'])**2).mean()