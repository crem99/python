import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === PARAMETRI ===
threshold = 20
grid_size = 10
max_k = 10  # numero massimo di cluster da testare
random_state = 42

# === GENERA DATI CON RANDOMIZZAZIONE ===
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=random_state)
noise = np.random.uniform(low=-0.5, high=0.5, size=X.shape)
X += noise  # randomizzazione

# === SCAN TEST ===
def has_clustering_structure(X, grid_size=10, threshold=20):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    cell_counts = np.zeros((grid_size, grid_size))
    for x, y in X:
        i = int((x - x_min) / (x_max - x_min) * grid_size)
        j = int((y - y_min) / (y_max - y_min) * grid_size)
        i = min(i, grid_size - 1)
        j = min(j, grid_size - 1)
        cell_counts[i, j] += 1

    max_count = cell_counts.max()
    print(f"Massimo numero di punti in una cella: {int(max_count)}")
    return max_count >= threshold

# === SSE ===
def compute_sse(X, labels, centroids):
    return sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(len(centroids)))

# === TROVA K OTTIMALE ===
def find_optimal_k(X, max_k=10):
    sse_list = []
    silhouette_list = []
    K_range = range(2, max_k+1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        sse = compute_sse(X, labels, centroids)
        sil_score = silhouette_score(X, labels)
        sse_list.append(sse)
        silhouette_list.append(sil_score)

    # Plot SSE e silhouette
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(K_range, sse_list, marker='o')
    ax[0].set_title('Metodo del gomito (SSE)')
    ax[0].set_xlabel('Numero di cluster k')
    ax[0].set_ylabel('SSE')

    ax[1].plot(K_range, silhouette_list, marker='s', color='green')
    ax[1].set_title('Silhouette score')
    ax[1].set_xlabel('Numero di cluster k')
    ax[1].set_ylabel('Silhouette')

    plt.tight_layout()
    plt.show()

    # k ottimale = massimo silhouette
    best_k = K_range[np.argmax(silhouette_list)]
    print(f"k ottimale scelto: {best_k}")
    return best_k

# === MAIN ===
if has_clustering_structure(X, grid_size=grid_size, threshold=threshold):
    print("Struttura di clustering rilevata. Cerco k ottimale...")

    best_k = find_optimal_k(X, max_k=max_k)

    kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    sse = compute_sse(X, labels, centroids)
    print(f"SSE finale: {sse:.2f}")

    # Plot finale
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
    plt.title(f'KMeans con k = {best_k} - SSE = {sse:.2f}')
    plt.show()

else:
    print("Nessuna struttura di clustering rilevata. Clustering non eseguito.")
