import numpy as np
import time
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")


def kmeans(data, K, n_iter=50):
    # Get first k distinct values
    distinct_values = set()
    counter = 0
    while len(distinct_values) < K and counter < len(data):
        distinct_values.add(data[counter])
        counter += 1

    # Initialize centroids and start the algorithm
    iter = 0
    old_centroids = []
    new_centroids = np.array(list(distinct_values))
    while iter < n_iter and set(old_centroids) != set(new_centroids):
        old_centroids = new_centroids

        sorted_centroids = sorted(old_centroids, reverse=True)
        C = vq(data, sorted_centroids)[0]

        new_centroids = [data[C == k].mean(axis=0) for k in range(K)]
        iter += 1

    return np.array(new_centroids), C


def kmeans2(data, K, n_iter=50):
    # Get first k distinct values
    distinct_values = set()
    counter = 0
    while len(distinct_values) < K and counter < len(data):
        distinct_values.add(data[counter])
        counter += 1

    # Initialize centroids and start the algorithm
    centroids = np.array(list(distinct_values))
    for iter in range(n_iter):
        # Cluster assignment
        sorted_centroids = sorted(centroids, reverse=True)
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in sorted_centroids]) for x_i in data])

        # Centroid calculation
        centroids = [data[C == k].mean(axis=0) for k in range(K)]

    return np.array(centroids), C


def kmeans3(data, K, n_iter=50):
    return KMeans(n_clusters=K, init=np.unique(data)[:K].reshape((-1, 1)), max_iter=n_iter).fit(data.reshape((-1, 1)))


def kmeans4(data, K, n_iter=50):
    distinct_values = set()
    counter = 0

    while len(distinct_values) < K and counter < len(data):
        distinct_values.add(data[counter])
        counter += 1

    tree = cKDTree(np.array(list(distinct_values)).reshape(-1, 1))

    for iter in range(n_iter):
        C = np.array(tree.query(data.reshape(-1, 1), k=1, p=1, n_jobs=1)[1])
        centroids = [data[C == k].mean(axis=0) for k in range(K)]
        tree = cKDTree(np.array(centroids).reshape(-1, 1))

clusters = [2, 5, 10]
dataset_sizes = [10**k for k in range(1, 5)]

for cluster in clusters:
    times_1 = []
    times_2 = []
    times_3 = []

    for dataset_size in dataset_sizes:
        x = np.random.rand(dataset_size)
        start = time.time()
        kmeans(x, cluster)
        times_1.append(time.time() - start)

        start = time.time()
        kmeans4(x, cluster)
        times_2.append(time.time() - start)

        start = time.time()
        kmeans3(x, cluster)
        times_3.append(time.time() - start)

    plt.xlabel('Dataset size')
    plt.ylabel('Execution time (s)')
    plt.plot(dataset_sizes, times_1, label='With VQ')
    plt.plot(dataset_sizes, times_2, label='Ball Tree')
    plt.plot(dataset_sizes, times_3, label='SKLearn')
    plt.title('Execution times in function of data set size for '+str(cluster)+' clusters')
    plt.legend()
    plt.show()

tests = [
         (np.random.rand(1e4), 'Clustering of uniformly distributed 1D array'),  # Uniform distribution,
         (np.random.normal(size=1e4), 'Clustering of standardnormal distributed 1D array')  # Normal distribution
        ]
for x, _title in tests:
    centers, Cluster = kmeans(x, 5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x,[1]*len(x),c=Cluster,s=50)
    for i in centers:
        ax.scatter(i,1,s=50,c='red',marker='+')
    ax.set_xlabel('x')
    plt.colorbar(scatter)
    plt.title(_title)
    plt.show()
