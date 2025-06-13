import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Carica il dataset del cancro al seno
data = datasets.load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names  # ['malignant', 'benign']

# Standardizza le feature: 
# le trasforma in modo che abbiano media = 0 e deviazione standard = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA per ridurre a 2D (per il grafico)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Dividi in train/test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
# Divide i dati in:
#Training set (80% dei dati)
#Test set (20%)
#random_state=42 rende la divisione ripetibile

# 5. Allena una SVM
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)
#Crea un modello SVM con kernel lineare, quindi cerca una retta (in 2D) che separi le classi.
#C=1.0 bilancia il margine e gli errori (è un parametro di regolarizzazione).
#.fit() allena il modello sui dati X_train e y_train

# 6. Predici e valuta
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))
#Usa il modello per predire le classi dei dati X_test.
#accuracy_score misura la percentuale di predizioni corrette.
#classification_report dà:
#Precision: quanti dei positivi predetti sono veramente positivi
#Recall: quanti dei veri positivi sono stati trovati
#F1-score: media armonica di precision e recall
#Support: numero di esempi per classe

#macro avg: Media semplice tra tutte le classi → non tiene conto della quantità di esempi. 
##Utile se le classi sono bilanciate

#weighted avg: Media ponderata per il numero di esempi in ogni classe → più rappresentativa se le classi sono sbilanciate.

# 7. Visualizza tutti i risultati in un'unica finestra
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
colors = ['red', 'blue']

# PCA 2D
for i, label in enumerate(np.unique(y)):
    axs[0].scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                   label=target_names[label], c=colors[i], alpha=0.6)
# Aggiungi vettori di supporto
sv = clf.support_vectors_
axs[0].scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
axs[0].set_xlabel('PCA 1')
axs[0].set_ylabel('PCA 2')
axs[0].set_title('SVM su dati biologici (cellule tumorali)')
axs[0].legend()
axs[0].grid(True)

# PCA 1 vs etichetta
for i, label in enumerate(np.unique(y)):
    axs[1].scatter(np.full(np.sum(y == label), i), X_pca[y == label, 0],
                   c=colors[i], alpha=0.6, label=target_names[label])
axs[1].set_xticks([0, 1])
axs[1].set_xticklabels(target_names)
axs[1].set_ylabel('PCA 1')
axs[1].set_title('Distribuzione PCA 1 per classe')
axs[1].legend()
axs[1].grid(True)

# PCA 2 vs etichetta
for i, label in enumerate(np.unique(y)):
    axs[2].scatter(np.full(np.sum(y == label), i), X_pca[y == label, 1],
                   c=colors[i], alpha=0.6, label=target_names[label])
axs[2].set_xticks([0, 1])
axs[2].set_xticklabels(target_names)
axs[2].set_ylabel('PCA 2')
axs[2].set_title('Distribuzione PCA 2 per classe')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()