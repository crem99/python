import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, silhouette_score

# === FUNZIONI DI SUPPORTO PER K-MEANS++ INIZIALIZZAZIONE ===
def _kmeans_plusplus_init(X, k, random_state):
    """
    Inizializzazione K-Means++ per i centroidi.
    Seleziona il primo centroide casualmente, poi gli altri basandosi sulla probabilità
    proporzionale al quadrato della distanza minima dai centroidi già scelti.
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    # 1. Scegli il primo centroide casualmente
    centroids = np.zeros((k, n_features))
    centroids[0] = X[rng.choice(n_samples)]

    # 2. Per ogni punto, calcola la distanza al centroide più vicino
    # 3. Scegli il prossimo centroide con probabilità proporzionale al quadrato della distanza
    for i in range(1, k):
        # Calcola le distanze di tutti i punti dai centroidi già scelti
        distances = np.array([np.min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) for x in X])
        
        # Le probabilità sono proporzionali al quadrato della distanza
        probabilities = distances / np.sum(distances)
        
        # Scegli il prossimo centroide basandosi su queste probabilità
        centroids[i] = X[rng.choice(n_samples, p=probabilities)]
    return centroids

def _kmeans_plusplus_init_weighted(X, k, w, random_state):
    """
    Inizializzazione K-Means++ pesata per i centroidi.
    Simile alla versione standard, ma usa la distanza pesata.
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)
    w_sqrt = np.sqrt(w) # Usiamo la radice quadrata dei pesi per la distanza euclidea pesata

    centroids = np.zeros((k, n_features))
    centroids[0] = X[rng.choice(n_samples)]

    for i in range(1, k):
        # Calcola le distanze pesate di tutti i punti dai centroidi già scelti
        distances = np.array([
            np.min([np.sum(w_sqrt * (x - c)**2) for c in centroids[:i]]) # Distanza euclidea pesata al quadrato
            for x in X
        ])
        
        probabilities = distances / np.sum(distances)
        centroids[i] = X[rng.choice(n_samples, p=probabilities)]
    return centroids

# === GENERA DATI CON RANDOMIZZAZIONE ===
# Ora generiamo sia il numero di campioni che i dati in modo casuale
rng = np.random.default_rng() # Inizializza un generatore di numeri casuali

# Genera un numero casuale di campioni tra 100 e 500 (inclusi)
n_samples = rng.integers(low=100, high=501) # high è esclusivo, quindi +1

n_features = 2 # Manteniamo 2 dimensioni per la visualizzazione

# Genera punti casuali con media 0 e deviazione standard 3
X = rng.normal(loc=0, scale=3, size=(n_samples, n_features))

# === FUNZIONE SSE ===
def compute_sse(X, labels, centroids):
    return sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(len(centroids)))

# === CLUSTERING ===
results = []

# --- KMeans++ (sklearn) ---
# n_init='auto' è un buon default per le versioni recenti di scikit-learn
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0, n_init='auto')
labels_km = kmeans.fit_predict(X)
sse_km = compute_sse(X, labels_km, kmeans.cluster_centers_)
silhouette_km = silhouette_score(X, labels_km)
results.append(('KMeans++', labels_km, kmeans.cluster_centers_, sse_km, silhouette_km))

# --- ISODATA (Stub) con K-Means++ inizializzazione ---
def isodata(X, initial_k=4, theta_n=10, theta_s=0.5, theta_c=1.0, max_iter=10, random_state=0):
    # Inizializzazione K-Means++
    centroids = _kmeans_plusplus_init(X, initial_k, random_state)

    for iteration in range(max_iter):
        # Assegna ogni punto al centroide più vicino
        labels = pairwise_distances_argmin(X, centroids)
        
        # Rimuovi cluster con pochi punti
        counts = np.array([np.sum(labels == i) for i in range(len(centroids))])
        
        # Mappa i vecchi indici dei cluster a quelli validi per ricalcolare i centroidi correttamente
        valid_cluster_indices = np.where(counts >= theta_n)[0]
        
        if len(valid_cluster_indices) == 0: # Tutti i cluster sono stati rimossi
            break
        
        new_centroids_list = []
        old_to_new_centroid_map = {}
        current_new_idx = 0
        for old_idx in valid_cluster_indices:
            cluster_points = X[labels == old_idx]
            if len(cluster_points) > 0:
                new_centroids_list.append(np.mean(cluster_points, axis=0))
                old_to_new_centroid_map[old_idx] = current_new_idx
                current_new_idx += 1
        
        if not new_centroids_list: # Nessun cluster valido dopo la rimozione
            centroids = np.array([])
            break
        
        centroids = np.array(new_centroids_list)
        
        # Aggiorna le labels per riflettere i cluster validi
        labels = pairwise_distances_argmin(X, centroids)

        # Split dei cluster con alta deviazione standard
        splits = []
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 1:
                std_dev = np.std(cluster_points, axis=0)
                if np.any(std_dev > theta_s):
                    direction = np.argmax(std_dev)
                    delta = 0.5 * std_dev[direction]
                    c1 = centroids[i].copy()
                    c2 = centroids[i].copy()
                    c1[direction] += delta
                    c2[direction] -= delta
                    splits.append(c1)
                    splits.append(c2)
                else:
                    splits.append(centroids[i])
            else: # Se un cluster ha 0 o 1 punto, non può essere splittato
                splits.append(centroids[i])
        centroids = np.array(splits)
        
        # Merge dei cluster troppo vicini
        merged = []
        used = set()
        # Per evitare errori se centroids è vuoto dopo uno split fallito o rimozione
        if len(centroids) == 0: 
            break

        for i in range(len(centroids)):
            if i in used:
                continue
            merged_this_iter = False
            for j in range(i + 1, len(centroids)):
                if j in used:
                    continue
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < theta_c:
                    new_c = (centroids[i] + centroids[j]) / 2
                    merged.append(new_c)
                    used.add(i)
                    used.add(j)
                    merged_this_iter = True
                    break
            if not merged_this_iter and i not in used:
                merged.append(centroids[i])
        
        # Assicurati che 'merged' non sia vuoto prima di convertirlo in un array NumPy
        if not merged:
            centroids = np.array([])
            break
        centroids = np.array(merged)

    # Ricalcola le labels finali per i centroidi risultanti
    if len(centroids) > 0:
        final_labels = pairwise_distances_argmin(X, centroids)
        sse = compute_sse(X, final_labels, centroids)
        # Calcola Silhouette Score solo se ci sono almeno 2 cluster
        if len(np.unique(final_labels)) > 1 and len(np.unique(final_labels)) < len(X) -1:
            silhouette = silhouette_score(X, final_labels)
        else:
            silhouette = np.nan # Impossibile calcolare se 0 o 1 cluster, o tutti i punti in un cluster
    else: # Nessun centroide rimasto
        final_labels = np.array([0] * len(X)) # Assegna tutti i punti a un cluster fittizio
        sse = np.nan
        silhouette = np.nan # Non calcolabile

    return final_labels, centroids, sse, silhouette

labels_iso, centroids_iso, sse_iso, silhouette_iso = isodata(X, initial_k=4, max_iter=10)
results.append(('ISODATA (stub)', labels_iso, centroids_iso, sse_iso, silhouette_iso))

# --- K-Means Generalizzato con Pesi con K-Means++ inizializzazione ---
def compute_weighted_distance(x, c, w):
    return np.sqrt(np.sum(w * (x - c) ** 2))

def assign_clusters_weighted(X, centroids, w):
    distances = np.array([
        [compute_weighted_distance(x, c, w) for c in centroids] for x in X
    ])
    return np.argmin(distances, axis=1)

def kmeans_generalizzato(X, k=4, w=None, max_iter=100, tol=1e-4, random_state=42):
    if w is None:
        w = np.ones(X.shape[1]) # pesi uniformi

    # Inizializzazione K-Means++ pesata
    centroids = _kmeans_plusplus_init_weighted(X, k, w, random_state)
    prev_sse = None

    for _ in range(max_iter):
        labels = assign_clusters_weighted(X, centroids, w)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
])  

        # Calcolo SSE pesato
        sse = sum(
            compute_weighted_distance(x, new_centroids[labels[i]], w) ** 2
            for i, x in enumerate(X)
)

        if prev_sse is not None and abs(prev_sse - sse) < tol:
             break

        centroids = new_centroids
        prev_sse = sse

    # Calcola Silhouette Score finale
    if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X) - 1:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = np.nan
        
    return labels, centroids, sse, silhouette

# Esegui il K-Means generalizzato con pesi
weights = np.array([1.0, 3.0]) # seconda feature più pesante
labels_gkm, centroids_gkm, sse_gkm, silhouette_gkm = kmeans_generalizzato(X, k=4, w=weights)
results.append(('K-Means Generalizzato', labels_gkm, centroids_gkm, sse_gkm, silhouette_gkm))

# === VISUALIZZAZIONE ===
plt.figure(figsize=(18, 5))
for i, (name, labels, centroids, sse, silhouette) in enumerate(results):
    plt.subplot(1, len(results), i+1)
    plt.title(f"{name}\nSSE={sse:.2f}\nSilhouette={silhouette:.3f}")

    # Assicurati che ci siano dati e centroidi validi per il plotting
    if X.shape[0] > 0 and len(np.unique(labels)) > 0:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=25)
        if len(centroids) > 0:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    else:
        # Se non ci sono cluster o dati validi, visualizza solo i punti in grigio
        plt.scatter(X[:, 0], X[:, 1], color='gray', s=25)

    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()