from utils import *

if __name__ == "__main__":
    """Question 1"""
    D2z = get_data("data/D2z.txt")
    plot_knn_predictions([-2,2,-2,2],[0.1,0.1],1,D2z)