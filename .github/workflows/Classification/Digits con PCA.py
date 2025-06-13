import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Funzione per calcolare l'errore di classificazione
def classification_error(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)

# Funzione per trovare le cifre più confuse in una matrice di confusione
def find_most_confused(cm, top_n=5):
    confused = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confused.append(((i, j), cm[i][j]))
    confused.sort(key=lambda x: x[1], reverse=True)
    return confused[:top_n]

# 1. Carica il dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2. Standardizza
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA per ridurre a 2D per visualizzazione
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# 5. SVM
svm = SVC(kernel="rbf", gamma="auto")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# 6. K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# 7. Report testuale
print("=== SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Errore:", classification_error(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

print("\n=== K-NN ===")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Errore:", classification_error(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# 8. Grafico delle predizioni
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_svm, cmap="tab10", alpha=0.7)
axs[0].set_title("SVM - Predizioni")
axs[0].set_xlabel("PCA 1")
axs[0].set_ylabel("PCA 2")
axs[0].grid(True)

axs[1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred_knn, cmap="tab10", alpha=0.7)
axs[1].set_title("K-NN - Predizioni")
axs[1].set_xlabel("PCA 1")
axs[1].set_ylabel("PCA 2")
axs[1].grid(True)

plt.suptitle("Confronto visivo tra SVM e K-NN sul dataset Digits (PCA 2D)", fontsize=14)
plt.tight_layout()
plt.show()

# 9. Matrici di confusione
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_knn = confusion_matrix(y_test, y_pred_knn)

ConfusionMatrixDisplay(cm_svm).plot(ax=axs[0], cmap="Blues", values_format='d')
axs[0].set_title("Matrice di Confusione - SVM")

ConfusionMatrixDisplay(cm_knn).plot(ax=axs[1], cmap="Greens", values_format='d')
axs[1].set_title("Matrice di Confusione - K-NN")

plt.suptitle("Confronto tra le Matrici di Confusione (SVM vs K-NN)", fontsize=14)
plt.tight_layout()
plt.show()

# 10. Cifre più confuse
print("\n--- Classi più confuse (SVM) ---")
for (i, j), count in find_most_confused(cm_svm):
    print(f"Cifra {i} scambiata per {j} → {count} volte")

print("\n--- Classi più confuse (K-NN) ---")
for (i, j), count in find_most_confused(cm_knn):
    print(f"Cifra {i} scambiata per {j} → {count} volte")
