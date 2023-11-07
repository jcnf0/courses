import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os 

def generate_synth(sigma):
    # Get means and covariances then sample
    mean1, mean2, mean3 = np.array([-1, -1]), np.array([1, -1]), np.array([0, 1])
    cov1, cov2, cov3 = np.array([[2*sigma, 0.5*sigma], [0.5*sigma, sigma]]), np.array([[sigma, -0.5*sigma], [-0.5*sigma, 2*sigma]]), np.array([[sigma, 0], [0, 2*sigma]])
    x1, x2, x3 = np.random.multivariate_normal(mean1, cov1, 100), np.random.multivariate_normal(mean2, cov2, 100), np.random.multivariate_normal(mean3, cov3, 100)

    return x1, x2, x3

def kmeans(data, k):
    def assign_clusters(data, centroids):
        # Get distances between data points and centroids
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # Return the index of the closest centroid
        return np.argmin(distances, axis=1)
    
    def update_centroids(data, clusters):
        # Initialize centroids
        centroids = np.zeros((k, data.shape[1]))
        
        # Iterate through clusters
        for i in range(k):
            # Get data points assigned to cluster i
            cluster_data = data[clusters == i]
            
            # Get mean of data points
            centroids[i] = np.mean(cluster_data, axis=0)
        
        return centroids
    # KMeans ++ initialization
    # Initialize centroids with one randomly chosen data point
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0])]
    
    # Iterate through the remaining centroids
    for i in range(1, k):
        # Get distances between data points and centroids
        distances = np.linalg.norm(data[:, np.newaxis] - centroids[:i], axis=2)
        
        # Get the minimum distance to any centroid for each data point
        min_distances = np.min(distances, axis=1)
        
        # Choose the next centroid with probability proportional to the square of the minimum distance
        probabilities = min_distances ** 2
        probabilities /= np.sum(probabilities)
        centroids[i] = data[np.random.choice(data.shape[0], p=probabilities)]
    
    # Iterate until convergence
    while True:
        # Assign clusters
        clusters = assign_clusters(data, centroids)
        
        # Update centroids
        new_centroids = update_centroids(data, clusters)
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        else:
            centroids = new_centroids
    
    # Initialize objective
    objective = 0
    
    # Iterate through clusters
    for i in range(centroids.shape[0]):
        # Get data points assigned to cluster i
        cluster_data = data[clusters == i]
        
        # Update objective
        objective += np.sum(np.linalg.norm(cluster_data - centroids[i], axis=1) ** 2)

    return clusters, centroids, objective

def expectation_maximization(data, k, num_iter=1000):
    n_samples, n_features = data.shape

    # Initialize the means, covariances, and mixing coefficients randomly
    means = np.random.rand(k, n_features)
    covs = np.array([np.cov(data.T) for _ in range(k)])
    weights = np.ones(k) / k

    for _ in range(num_iter):
        # E-step
        gamma = np.zeros((data.shape[0], k))
        for i in range(k):
            gamma[:, i] = weights[i] * multivariate_normal.pdf(data, means[i], covs[i])
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        # M-step
        for i in range(k):
            gamma_i = gamma[:, i]
            total_gamma_i = gamma_i.sum()
            means[i] = np.dot(gamma_i, data) / total_gamma_i
            covs[i] = np.dot(gamma_i * (data - means[i]).T, data - means[i]) / total_gamma_i
            weights[i] = total_gamma_i / data.shape[0]

    # Predict the cluster assignments
    gamma = np.zeros((data.shape[0], k))
    for i in range(k):
        gamma[:, i] = weights[i] * multivariate_normal.pdf(data, means[i], covs[i])
    clusters = np.argmax(gamma, axis=1)

    objective = np.sum(np.log(np.sum(gamma, axis=1)))
    return clusters, weights, objective

def clustering_accuracy(clusters):
    # Initialize accuracy
    # Get the best permutation of labels
    accuracy = 0
    labels1 = np.concatenate((np.zeros(100), np.ones(100), np.ones(100) * 2))
    labels2 = np.concatenate((np.zeros(100), np.ones(100) * 2, np.ones(100)))
    labels3 = np.concatenate((np.ones(100), np.zeros(100), np.ones(100) * 2))
    labels4 = np.concatenate((np.ones(100), np.zeros(100) * 2, np.ones(100)))
    labels5 = np.concatenate((np.ones(100) * 2, np.zeros(100), np.ones(100)))
    labels6 = np.concatenate((np.ones(100) * 2, np.zeros(100) * 2, np.ones(100)))
    labels = [labels1, labels2, labels3, labels4, labels5, labels6]

    for label in labels:
        accuracy = max(accuracy, np.sum(label == clusters) / label.shape[0])  

    return accuracy

def get_data(file):
    data = np.genfromtxt(file, delimiter=',') 
    return data

def buggy_pca(data, d):    
    # Get SVD of data
    _, _, V = np.linalg.svd(data)
    
    # Get top d right singular vectors
    top_d = V[:d].T

    # Get reconstructed data and error
    reconstructed_data = np.dot(top_d, np.dot(top_d.T, data.T)).T
    # Error is the average square difference between the original data and the reconstructed data
    error = np.sum(np.linalg.norm(data-reconstructed_data, axis=1)**2)
    
    return top_d, error, reconstructed_data

def demeaned_pca(data, d):
    # Substract the mean from data before performing PCA
    mean = np.mean(data, axis=0)
    X = data - mean

    # Get SVD of data
    _, _, V = np.linalg.svd(X)
    
    # Get top d right singular vectors
    top_d = V[:d].T

    # Get reconstructed data and error
    reconstructed_data = np.dot(top_d, np.dot(top_d.T, X.T)).T
    reconstructed_data = reconstructed_data + mean
    # Error is the average square differnce between the original data and the reconstructed data
    error = np.sum(np.linalg.norm(data-reconstructed_data, axis=1)**2)
    
    return top_d, error, reconstructed_data

def normalized_pca(data, d):
    # Normalize data before performing PCA
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    X = (data-mean) / std

    # Get SVD of data
    _, _, V = np.linalg.svd(X)
    
    # Get top d right singular vectors
    top_d = V[:d].T

    # Get reconstructed data and error
    reconstructed_data = np.dot(top_d, np.dot(top_d.T, X.T)).T
    reconstructed_data = reconstructed_data * std + mean
    # Error is the average square difference between the original data and the reconstructed data
    error = np.sum(np.linalg.norm(data-reconstructed_data, axis=1)**2)

    return top_d, error, reconstructed_data

def dro(data,d):
    # Get the mean of the data
    b = np.mean(data, axis=0)

    # Get SVD of data without mean
    U, S, V = np.linalg.svd(data-b)

    # Get the first d columns of U
    U_d = U[:,:d]

    # Get the first d rows of V
    V_d = V[:d,:]

    # Get the first d columns and rows of S
    S_d = np.diag(S[:d])

    # Compute Z=U_d * S_d and A = V_d
    Z = np.dot(U_d, S_d)
    A = V_d.T
    
    # Get reconstructed data and error we perform A*x for all x in data
    reconstructed_data = np.dot(A, np.dot(A.T, (data-b).T)) + np.dot(b.reshape(-1,1), np.ones((1,data.shape[0])))
    reconstructed_data = reconstructed_data.T 
    # Error is the average square difference between the original data and the reconstructed data
    error = np.sum(np.linalg.norm(data-reconstructed_data, axis=1)**2)

    return Z, A, error, reconstructed_data

def compare_2d(data,reconstructed,save_path = None):
    # Plot original data
    plt.scatter(data[:,0], data[:,1], label='Original data')
    
    # Plot reconstructed data
    plt.scatter(reconstructed[:,0], reconstructed[:,1], label='Reconstructed data')
    
    plt.legend()

    # Check if dir exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.clf()