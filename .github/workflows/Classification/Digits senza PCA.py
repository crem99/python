import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Funzione per trovare le cifre più confuse
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
X = digits.data  # 64 features (immagini 8x8)
y = digits.target

# 2. Standardizza i dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Allena SVM
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# 5. Allena K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# 6. Report e valutazioni
print("=== SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Errore:", classification_error(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

print("\n=== K-NN ===")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Errore:", classification_error(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# 7. Matrici di confusione
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_knn = confusion_matrix(y_test, y_pred_knn)

ConfusionMatrixDisplay(cm_svm).plot(ax=axs[0], cmap="Blues", values_format='d')
axs[0].set_title("Matrice di Confusione - SVM")

ConfusionMatrixDisplay(cm_knn).plot(ax=axs[1], cmap="Greens", values_format='d')
axs[1].set_title("Matrice di Confusione - K-NN")

plt.suptitle("Confronto tra SVM e K-NN (con tutte le 64 feature)", fontsize=14)
plt.tight_layout()
plt.show()

# 8. Cifre più confuse
print("\n--- Classi più confuse (SVM) ---")
for (i, j), count in find_most_confused(cm_svm):
    print(f"Cifra {i} scambiata per {j} → {count} volte")

print("\n--- Classi più confuse (K-NN) ---")
for (i, j), count in find_most_confused(cm_knn):
    print(f"Cifra {i} scambiata per {j} → {count} volte")
