import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler # Per normalizzare i dati

# === FUNZIONI DI SUPPORTO PER K-MEANS++ INIZIALIZZAZIONE (dal codice precedente) ===
def _kmeans_plusplus_init(X, k, random_state):
    """
    Inizializzazione K-Means++ per i centroidi.
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    centroids = np.zeros((k, n_features))
    centroids[0] = X[rng.choice(n_samples)]

    for i in range(1, k):
        distances = np.array([np.min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) for x in X])
        probabilities = distances / np.sum(distances)
        centroids[i] = X[rng.choice(n_samples, p=probabilities)]
    return centroids

def _kmeans_plusplus_init_weighted(X, k, w, random_state):
    """
    Inizializzazione K-Means++ pesata per i centroidi. (Non strettamente usata qui ma mantenuta)
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)
    w_sqrt = np.sqrt(w)

    centroids = np.zeros((k, n_features))
    centroids[0] = X[rng.choice(n_samples)]

    for i in range(1, k):
        distances = np.array([
            np.min([np.sum(w_sqrt * (x - c)**2) for c in centroids[:i]])
            for x in X
        ])
        probabilities = distances / np.sum(distances)
        centroids[i] = X[rng.choice(n_samples, p=probabilities)]
    return centroids

# === FUNZIONE SSE ===
def compute_sse(X, labels, centroids):
    return sum(np.sum((X[labels == i] - centroids[i])**2) for i in range(len(centroids)))

# === SIMULAZIONE DATI DI ESPRESSIONE GENICA ===
# 500 geni, 5 condizioni sperimentali
n_genes = 500
n_conditions = 5
n_true_clusters = 5 # Supponiamo ci siano 5 pattern di espressione "veri"

# Generazione di pattern di espressione "base" per ogni cluster
# Ogni riga è un pattern di espressione attraverso le 5 condizioni
true_patterns = np.array([
    [1.0, 2.0, 3.0, 2.5, 1.5], # Cluster 1: Espressione crescente e poi decrescente
    [5.0, 4.0, 3.0, 2.0, 1.0], # Cluster 2: Espressione decrescente
    [2.0, 2.0, 2.0, 2.0, 2.0], # Cluster 3: Espressione costante
    [0.5, 1.0, 1.5, 2.0, 2.5], # Cluster 4: Espressione sempre crescente
    [3.0, 1.0, 3.0, 1.0, 3.0]  # Cluster 5: Espressione fluttuante
])

# Assegna ogni gene a un "vero" cluster e aggiungi rumore
gene_expression_data = np.zeros((n_genes, n_conditions))
true_gene_clusters = np.random.randint(0, n_true_clusters, n_genes)

for i in range(n_genes):
    pattern = true_patterns[true_gene_clusters[i]]
    # Aggiungi rumore gaussiano all'espressione
    gene_expression_data[i] = pattern + np.random.normal(0, 0.3, n_conditions)

# --- PRE-ELABORAZIONE: SCALING DEI DATI ---
# È cruciale scalare i dati di espressione genica, specialmente se le scale di misura variano.
# StandardScaler porta la media a 0 e la deviazione standard a 1 per ogni condizione (feature).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(gene_expression_data)

print(f"Dimensione del dataset (Geni x Condizioni): {X_scaled.shape}")
print(f"Prime 5 righe dei dati scalati:\n{X_scaled[:5, :]}")

# === CLUSTERING K-MEANS sui dati di espressione genica ===
# Usiamo K-Means con l'inizializzazione K-Means++ che abbiamo definito
# Per un esempio reale, spesso si prova un range di k e si usa il metodo del gomito/silhouette
# qui scegliamo k=n_true_clusters (5) per confrontare.
k_optimal = n_true_clusters # Numero di cluster che ci aspettiamo di trovare

kmeans_bio = KMeans(n_clusters=k_optimal, init=_kmeans_plusplus_init, n_init=1, random_state=42)
# Nota: passando una funzione a init, n_init deve essere 1.
# Per un uso più robusto di scikit-learn con la sua init 'k-means++', si userebbe:
# kmeans_bio = KMeans(n_clusters=k_optimal, init='k-means++', n_init='auto', random_state=42)

labels_bio = kmeans_bio.fit_predict(X_scaled)
centroids_bio = kmeans_bio.cluster_centers_
sse_bio = compute_sse(X_scaled, labels_bio, centroids_bio)

# Calcola Silhouette Score
if len(np.unique(labels_bio)) > 1 and len(np.unique(labels_bio)) < len(X_scaled) - 1:
    silhouette_bio = silhouette_score(X_scaled, labels_bio)
else:
    silhouette_bio = np.nan

print(f"\nRisultati K-Means per i dati di espressione genica:")
print(f"Numero di cluster trovati: {len(np.unique(labels_bio))}")
print(f"SSE: {sse_bio:.2f}")
print(f"Silhouette Score: {silhouette_bio:.3f}")

# === VISUALIZZAZIONE DEI PATTERN DI ESPRESSIONE PER OGNI CLUSTER ===
plt.figure(figsize=(12, 6))

# Visualizza i pattern di espressione medi per ogni cluster
for i in range(len(centroids_bio)):
    cluster_points = X_scaled[labels_bio == i]
    
    # Calcola il pattern medio del cluster (il centroide)
    mean_pattern = centroids_bio[i]
    
    # Esempio: visualizza solo un sottoinsieme di geni per non sovrapporre troppo
    # o visualizza solo il centroide per chiarezza
    
    # Plotta i singoli geni del cluster (opzionale, può essere denso)
    # for gene_idx in range(min(20, len(cluster_points))): # Plotta max 20 geni per cluster
    #     plt.plot(range(1, n_conditions + 1), cluster_points[gene_idx], alpha=0.1, color=plt.cm.viridis(i/len(centroids_bio)))

    plt.plot(range(1, n_conditions + 1), mean_pattern, 
             label=f'Cluster {i} (N={len(cluster_points)})',
             linewidth=2, marker='o', linestyle='-',
             color=plt.cm.tab10(i)) # Utilizza una colormap per i cluster

plt.title('Pattern di Espressione Genica Medi per Cluster')
plt.xlabel('Condizione Sperimentale')
plt.ylabel('Livello di Espressione (Scalato)')
plt.xticks(range(1, n_conditions + 1), [f'Cond {j+1}' for j in range(n_conditions)])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# === ANALISI AGGIUNTIVA: CONFRONTO CON I CLUSTER VERI (se conosciuti) ===
# Dato che abbiamo generato i dati, conosciamo i "veri" cluster.
# Questo passaggio non sarebbe possibile con dati reali non etichettati.
from sklearn.metrics import adjusted_rand_score

if 'true_gene_clusters' in locals():
    # Poiché l'assegnazione dei cluster di K-Means è arbitraria (cluster 0 potrebbe corrispondere al cluster 2 vero),
    # usiamo metriche che non dipendono dall'ordine delle etichette.
    ari = adjusted_rand_score(true_gene_clusters, labels_bio)
    print(f"\nAdjusted Rand Index (confronto con cluster veri): {ari:.3f}")

    # Visualizzazione dei centroidi trovati vs. pattern veri per una migliore comprensione
    plt.figure(figsize=(12, 6))
    for i in range(len(true_patterns)):
        plt.plot(range(1, n_conditions + 1), scaler.transform(true_patterns[i].reshape(1, -1))[0], 
                 label=f'True Pattern {i}', linestyle='--', alpha=0.7, color=plt.cm.tab10(i))
    
    for i in range(len(centroids_bio)):
        plt.plot(range(1, n_conditions + 1), centroids_bio[i], 
                 label=f'Found Centroid {i}', linestyle='-', marker='x', linewidth=2, color=plt.cm.tab10(i))
    
    plt.title('Confronto tra Pattern di Espressione Veri e Centroidi Trovati')
    plt.xlabel('Condizione Sperimentale')
    plt.ylabel('Livello di Espressione (Scalato)')
    plt.xticks(range(1, n_conditions + 1), [f'Cond {j+1}' for j in range(n_conditions)])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()