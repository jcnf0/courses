from utils import *

if __name__ == "__main__":

    """Question 1"""
    print("Question 1")
    D2z = get_data("data/D2z.txt")
    plot_knn_predictions([-2,2,-2,2], [0.1,0.1], 1, D2z, output_file="figures/knn_predictions.pdf")

    """Question 2"""
    print("Question 2")
    emails = pd.read_csv("data/emails.csv").drop("Email No.", axis=1)
    knn_cross_val(1, emails, "Prediction", 5, output_file="figures/Q1_knn_cross_val.csv")

    """Question 3"""
    print("Question 3")
    logistic_regression_cross_val(emails, 0.001, "Prediction", 5)

    """Question 4"""
    print("Question 4")
    k_list = [1, 3, 5, 7, 10]
    mean_accuracy_list = []
    for k in [1, 3, 5, 7, 10]:
        print("k = {}".format(k))
        mean_accuracy, mean_precision, mean_recall = knn_cross_val(k, emails, "Prediction", 5, output_file="figures/Q4_{}nn_cross_val.csv".format(k))
        mean_accuracy_list.append(mean_accuracy)

    plt.plot(k_list, mean_accuracy_list)
    plt.xlabel("k")
    plt.ylabel("Mean accuracy")    
    plt.show()
    plt.clf()
    
    """Question 5"""
    print("Question 5")
    # 4000/1000 split
    training_data = emails.iloc[:4000]
    test_data = emails.iloc[4000:]

    roc_curves(5, training_data, test_data, "Prediction", eta=0.001, output_file="figures/roc_curves.pdf")

    