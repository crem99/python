from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# 1. Genera dati sintetici
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. Definisci i modelli
models = {
    'KMeans': KMeans(n_clusters=4),
    'AgglomerativeClustering': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}

#------------
range_n_clusters = range(2, 10)
silhouette_scores = []
calinski_scores = []
davies_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)
    
    silhouette_scores.append(silhouette_score(X, labels))
    calinski_scores.append(calinski_harabasz_score(X, labels))
    davies_scores.append(davies_bouldin_score(X, labels))
#---------------


# 3. Applica clustering, calcola silhouette score e visualizza
plt.figure(figsize=(15,4))

for i, (name, model) in enumerate(models.items()):
    if name == 'DBSCAN':
        # DBSCAN non necessita di fit_predict separato
        labels = model.fit_predict(X)
    else:
        model.fit(X)
        labels = model.labels_
    
    # Calcola silhouette score (ignora i rumori DBSCAN con label = -1)
    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(X, labels)
    else:
        # se DBSCAN trova rumore, escludi i -1
        mask = labels != -1
        if sum(mask) > 1 and len(set(labels[mask])) > 1:
            score = silhouette_score(X[mask], labels[mask])
        else:
            score = float('nan')  # non calcolabile
    
    print(f"{name} - silhouette score: {score:.3f}")
    
    # Plot
    plt.subplot(1, 3, i+1)
    plt.title(f"{name}\nSilhouette score: {score:.3f}")
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    if name == 'KMeans':
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=200, c='red', marker='x')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()
