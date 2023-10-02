from utils import *
import csv
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':

    """Question 1"""

    """Question 2"""
    dcraft = get_data("data/Dcraft.txt")
    treecraft = DecisionTree()
    treecraft.fit(dcraft)
    treecraft.plot_tree()
    show_data(dcraft, "figures/scatter_plots/Dcraft.pdf", size=50,limits=[-1,2,-1,2])

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
    show_data(d1, "figures/scatter_plots/D1.pdf")
    show_data(d2, "figures/scatter_plots/D2.pdf")

    tree1 = DecisionTree()
    tree1.fit(d1)
    tree1.plot_tree()

    tree2 = DecisionTree()
    tree2.fit(d2)
    tree2.plot_tree()

    # Plot decision boundaries
    plot_decision_boundary(tree1, d1, "figures/decision_boundaries/Decision_D1.pdf",[0,1,0,1])
    plot_decision_boundary(tree2, d2, "figures/decision_boundaries/Decision_D2.pdf",[0,1,0,1])

    """Question 7"""
    sizes = [32, 128, 512, 2048, 8192]
    train_sets, test_set = nested_sets("data/Dbig.txt")
    trees = [DecisionTree() for k in range(len(train_sets))]

    for k in range(len(train_sets)):
        trees[k].fit(train_sets[k])
        plot_decision_boundary(trees[k],train_sets[k], "figures/decision_boundaries/Decision_Dbig_{}.pdf".format(sizes[k]),[-1.5,1.5,-1.5,1.5])
        
    # Show n num_nodes and err_n
    get_n_num_nodes_errn(trees, sizes, test_set, output_file= "figures/error_n/n_nodes_errn.csv")
    
    # Plot n vs errn
    n_list, errn_list = plot_error_n(trees, sizes, test_set, output_file= "figures/error_n/Decision_Dbig.pdf")
        

    #Create CSV file with number of nodes and err_n
    with open('Decision_Dbig.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'err_n'])
        writer.writerows(zip(n_list, errn_list))

    
    """Scikit Learn"""
    sktrees = [DecisionTreeClassifier() for k in range(len(train_sets))]

    for k in range(len(train_sets)):
        sktrees[k].fit(train_sets[k][['x1','x2']],train_sets[k]['y'])

    # Show n num_nodes and err_n
    get_n_num_nodes_errn(sktrees, sizes, test_set, label="sklearn", output_file= "figures/error_n/n_nodes_errn_sklearn.csv")

    # Plot n vs errn
    skn, skerrn_list = plot_error_n(sktrees, sizes, test_set, label="sklearn", output_file= "figures/error_n/skDecision_Dbig.pdf")

    #Create CSV file with number of nodes and err_n
    with open('skDecision_Dbig.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'err_n'])
        writer.writerows(zip(skn, skerrn_list))

    """Lagrange Interpolation"""
    a = 0
    b = 1
    n = 100

    lag_train = sample_uniform(a,b,n)
    lag_train_noisy = lag_train.copy()
    lag_test = sample_uniform(a,b,n)
    train_err, test_err = lagrange_test(lag_train, lag_test)

    plot_lagrange(lag_train, lag_test, "figures/lagrange/lagrange.pdf")

    # Test for different values of e_std
    e_std_list = [0, 0.1, 0.5, 1, 2, 5, 10]
    train_err_list = [train_err]
    test_err_list = [test_err]

    for e_std in e_std_list:
        # Add noise to train set x
        lag_train_noisy['x'] = lag_train['x'] + np.random.normal(0, e_std, n)
        lag_train_noisy['y'] = np.sin(lag_train_noisy['x'])
        train_err, test_err = lagrange_test(lag_train_noisy, lag_test)
        train_err_list.append(train_err)
        test_err_list.append(test_err)
        plot_lagrange(lag_train_noisy, lag_test, "figures/lagrange/lagrange_std{}.pdf".format(e_std))

    with open('lagrange.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['std', 'train_error', 'test_error'])
        writer.writerows(zip(e_std_list, train_err_list, test_err_list))