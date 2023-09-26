from utils import *
import csv
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':

    # """Question 1"""

    # """Question 2"""

    """Question 3"""
    druns = get_data("data/Druns.txt")
    treeruns = DecisionTree()
    treeruns.fit(druns)

    """Question 4"""
    d3leaves = get_data("data/D3leaves.txt")
    tree3leaves = DecisionTree()
    tree3leaves.fit(d3leaves) 
    tree3leaves.plot_tree()

    """Question 5"""

    """Question 6"""
    d1 = get_data("data/D1.txt")
    d2 = get_data("data/D2.txt")

    # Plot data
    show_data(d1, "cs-760/homework-2/figures/scatter_plots/D1.pdf")
    show_data(d2, "cs-760/homework-2/figures/scatter_plots/D2.pdf")

    tree1 = DecisionTree()
    tree1.fit(d1)
    tree1.plot_tree()

    tree2 = DecisionTree()
    tree2.fit(d2)
    tree2.plot_tree()

    # Plot decision boundaries
    plot_decision_boundary(tree1, d1, "cs-760/homework-2/figures/decision_boundaries/Decision_D1.pdf",[0,1,0,1])
    plot_decision_boundary(tree2, d2, "cs-760/homework-2/figures/decision_boundaries/Decision_D2.pdf",[0,1,0,1])

    """Question 7"""
    sizes = [32, 128, 512, 2048, 8192]
    train_sets, test_set = nested_sets("data/Dbig.txt")
    trees = [DecisionTree() for k in range(len(train_sets))]

    for k in range(len(train_sets)):
        trees[k].fit(train_sets[k])
        plot_decision_boundary(trees[k],train_sets[k], "cs-760/homework-2/figures/decision_boundaries/Decision_Dbig_{}.pdf".format(sizes[k]),[-1.5,1.5,-1.5,1.5])
        
    # Plot n vs errn
    num_nodes_list, errn_list = plot_error_n(trees, test_set, output_file= "cs-760/homework-2/figures/error_n/Decision_Dbig.pdf")
        

    #Create CSV file with number of nodes and err_n
    with open('Decision_Dbig.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['num_nodes', 'err_n'])
        writer.writerows(zip(num_nodes_list, errn_list))

    
    """Scikit Learn"""
    sktrees = [DecisionTreeClassifier() for k in range(len(train_sets))]

    for k in range(len(train_sets)):
        sktrees[k].fit(train_sets[k][['x1','x2']],train_sets[k]['y'])

    # Plot n vs errn
    sknum_nodes_list, skerrn_list = plot_error_n(sktrees, test_set, label="sklearn", output_file= "cs-760/homework-2/figures/error_n/skDecision_Dbig.pdf")

    #Create CSV file with number of nodes and err_n
    with open('skDecision_Dbig.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['num_nodes', 'err_n'])
        writer.writerows(zip(sknum_nodes_list, skerrn_list))

    """Lagrange Interpolation"""
    a = 0
    b = 1
    n = 100
    e_mean = 0
    e_std = 1

    lag_train = sample_uniform(a,b,n,0,1)
    lag_test = sample_uniform(a,b,n,0,1)

    print(lagrange_test(lag_train, lag_test))

    # Test for different values of e_std
    e_std_list = [0.1, 0.5, 1, 2, 5, 10]
    train_err_list = []
    test_err_list = []

    for e_std in e_std_list:
        lag_train = sample_uniform(a,b,n,e_mean,e_std)
        lag_test = sample_uniform(a,b,n,e_mean,e_std)
        train_err, test_err = lagrange_test(lag_train, lag_test)
        train_err_list.append(train_err)
        test_err_list.append(test_err)

    with open('lagrange.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['std', 'train_error', 'test_error'])
        writer.writerows(zip(e_std_list, train_err_list, test_err_list))