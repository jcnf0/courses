from utils import *

if __name__ == '__main__':
    data2D = get_data('data/data2D.csv')
    d = 1
    top_d1, error1, recon1 = buggy_pca(data2D, d)
    top_d2, error2, recon2= demeaned_pca(data2D, d)
    top_d3, error3, recon3 = normalized_pca(data2D, d)
    _,_, error4, recon4 = dro(data2D, d)

    compare_2d(data2D, recon1, 'figures/buggy_pca2D.pdf')
    compare_2d(data2D, recon2, 'figures/demeaned_pca2D.pdf')
    compare_2d(data2D, recon3, 'figures/normalized_pca2D.pdf')
    compare_2d(data2D, recon4, 'figures/dro2D.pdf')
    print("Error for buggy PCA: {}".format(error1))
    print("Error for demeaned PCA: {}".format(error2))
    print("Error for normalized PCA: {}".format(error3))
    print("Error for DRO: {}".format(error4))

    # Plot errors for 2D
    plt.plot([error1, error2, error3, error4])
    plt.xticks([0, 1, 2, 3], ['Buggy PCA', 'Demeaned PCA', 'Normalized PCA', 'DRO'])
    plt.title('Error for 2D')
    plt.ylabel('Error')
    plt.savefig("figures/error_2D.pdf")

    data1000D = get_data('data/data1000D.csv')
    Z, A, error, reconstructed_data = dro(data1000D, 50)
    print(error)
    
    error_list_buggy = []
    error_list_demeaned = []
    error_list_normalized = []
    error_list_dro = []
    for d in [5, 10, 25, 50, 100 , 200, 500]:
        _, error1,_ = buggy_pca(data1000D, d)
        _, error2,_ = demeaned_pca(data1000D, d)
        _, error3,_ = normalized_pca(data1000D, d)
        _,_,error4,_ = dro(data1000D, d)


        error_list_buggy.append(error1)
        error_list_demeaned.append(error2)
        error_list_normalized.append(error3)
        error_list_dro.append(error4)
    
    plt.plot(error_list_buggy, label='Buggy PCA')
    plt.plot(error_list_demeaned, label='Demeaned PCA')
    plt.plot(error_list_normalized, label='Normalized PCA')
    plt.plot(error_list_dro, label='DRO')
    plt.legend()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], [5, 10, 25, 50, 100 , 200, 500, 1000])
    plt.title('Error vs d')
    plt.xlabel('d')
    plt.ylabel('Error')
    plt.savefig("figures/error_vs_d.pdf")
    plt.clf()
    print("Error for buggy PCA: {}".format(error_list_buggy))
    print("Error for demeaned PCA: {}".format(error_list_demeaned))
    print("Error for normalized PCA: {}".format(error_list_normalized))
    print("Error for DRO: {}".format(error_list_dro))
    