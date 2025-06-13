import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Funzione per salvare un modello
def salva_modello(modello, nome):
    if not os.path.exists("modelli_salvati"):
        os.makedirs("modelli_salvati")
    safe_name = nome.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
    filename = f"{safe_name}_model.pkl"

    full_path = os.path.join("modelli_salvati", filename)

    joblib.dump(modello, full_path)
    print(f"✅ Modello salvato in: {os.path.abspath(full_path)}")

    if os.path.exists(filename):
        print(f"✅ Verificato: il file {filename} è stato creato correttamente.")
    else:
        print(f"❌ Errore: il file {filename} non è stato trovato.")

    print("📂 Cartella di lavoro:", os.getcwd())

# 1. Carica e standardizza il dataset
digits = load_digits()
X = digits.data
y = digits.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Divisione in train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Insieme dei modelli da confrontare
models = {
    "SVM (RBF)": SVC(kernel='rbf', gamma='scale'),
    "K-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "MLP (Neural Net)": MLPClassifier(max_iter=1000)
}

# 4. Setup dei grafici multipli
fig, axs = plt.subplots(2, 4, figsize=(18, 9))
axs = axs.flatten()

best_models = []

print("=== Confronto tra modelli (con cross-validation) ===\n")

# 5. Valuta e salva i modelli
for i, (name, model) in enumerate(models.items()):
    print(f"🔎 Modello: {name}")
    
    # Cross-validation
    scores = cross_val_score(model, X_scaled, y, cv=5)
    mean_score = scores.mean()
    print(f"  ➤ Accuratezze cross-validation: {np.round(scores, 3)}")
    print(f"  ➤ Accuracy media: {mean_score:.4f}")

    # Addestramento e predizione sul test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("  ➤ Classification report:")
    print(classification_report(y_test, y_pred))

    # Salva il modello se la media è almeno 0.94
    if mean_score >= 0.94:
        salva_modello(model, name)
        best_models.append((name, mean_score))
        print(f"✅ Modello '{name}' salvato (accuracy: {mean_score:.4f})\n")

    # Matrice di confusione nel grafico
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=axs[i], cmap='Blues', colorbar=False)
    axs[i].set_title(f"{name}\nAccuracy media: {mean_score:.2%}")
    axs[i].grid(False)

# Nascondi subplot vuoti se presenti
for j in range(i + 1, len(axs)):
    axs[j].axis('off')

plt.suptitle("Confronto tra modelli - Matrici di Confusione", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 6. Stampa dei migliori modelli salvati
if best_models:
    print("\n📌 Migliori modelli salvati (accuracy >= 94%):")
    for name, score in best_models:
        print(f"✔️ {name} → Accuracy media: {score:.4f}")
else:
    print("\n⚠️ Nessun modello ha raggiunto il 94% di accuratezza media.")


