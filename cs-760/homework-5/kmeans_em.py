from utils import *

if __name__ == '__main__':
    # Create synthetic data
    num_restart = 50

    accuracy_list_kmeans = []
    accuracy_list_em = []
    objective_list_kmeans = []
    objective_list_em = []
    for sigma in [0.5, 1, 2, 4, 8]:
        print("\nSIGMA = {}".format(sigma))
        x1,x2,x3 = generate_synth(sigma)
        
        data = np.concatenate((x1, x2, x3))
        plt.plot(x1[:,0], x1[:,1], 'ro', label='x1')
        plt.plot(x2[:,0], x2[:,1], 'bo', label='x2')
        plt.plot(x3[:,0], x3[:,1], 'go', label='x3')
        plt.legend()
        plt.title('Synthetic data')
        plt.savefig("figures/synthetic_data_sigma{}.pdf".format(sigma))
        plt.clf()
        # K means on x1 x2 and x3 
        clusters_kmeans,centroids_kmeans,objective_kmeans = kmeans(data, 3)
        # EM on x1 x2 and x3
        clusters_em, weights_em, objective_em = expectation_maximization(data, 3)

        # Get labels from the concatenated data

        accuracy_kmeans = clustering_accuracy(clusters_kmeans)
        accuracy_em = clustering_accuracy(clusters_em)
        
        for k in range(num_restart):
            new_clusters_kmeans,new_centroids_kmeans, new_objective_kmeans = kmeans(data, 3)
            new_clusters_em, new_weights_em, new_objective_em = expectation_maximization(data, 3)
            
            if new_objective_kmeans < objective_kmeans:
                objective_kmeans = new_objective_kmeans
                clusters_kmeans = new_clusters_kmeans
                centroids_kmeans = new_centroids_kmeans

            if new_objective_em < objective_em:
                objective_em = new_objective_em
                clusters_em = new_clusters_em
                weights_em = new_weights_em

        accuracy_list_kmeans.append(accuracy_kmeans)
        accuracy_list_em.append(accuracy_em)
        objective_list_kmeans.append(objective_kmeans)
        objective_list_em.append(objective_em)
        print("K-means accuracy : {}%    Objective : {}".format(accuracy_kmeans*100,int(objective_kmeans)))
        print("EM accuracy : {}%    Objective : {}".format(accuracy_em*100,int(objective_em)))

    plt.plot(accuracy_list_em, label='EM')
    plt.plot(accuracy_list_kmeans, label='K-means')

    # Change name of x ticks
    plt.xticks([0,1,2,3,4], [0.5,1,2,4,8])
    plt.xlabel('Sigma')
    plt.ylabel('Kmeans/EM accuracy')
    plt.legend()
    plt.title('Kmeans and EM vs Sigma')
    plt.savefig("figures/accuracy_vs_sigma.pdf")
    plt.clf()

    # Plot objective vs sigma
    plt.plot(objective_list_em, label='EM')
    plt.plot(objective_list_kmeans, label='K-means')

    # Change name of x ticks
    plt.xticks([0,1,2,3,4], [0.5,1,2,4,8])
    plt.xlabel('Sigma')
    plt.ylabel('Kmeans/EM objective')
    plt.legend()
    plt.title('Kmeans and EM vs Sigma')
    plt.savefig("figures/objective_vs_sigma.pdf")
    plt.clf()
