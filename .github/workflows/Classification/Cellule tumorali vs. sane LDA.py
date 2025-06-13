import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report

# 1. Carica il dataset
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names  # ['malignant', 'benign']

# 2. Standardizza le feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Applica LDA (1 componente perché ci sono solo 2 classi)
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

# 4. Suddivide in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# 5. Allena un classificatore LDA
clf = LDA()
clf.fit(X_train, y_train)

# 6. Predizione e valutazione
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))

# 7. Grafico: distribuzione dei dati ridotti su una dimensione
plt.figure(figsize=(8, 5))
colors = ['red', 'blue']
for label, color, name in zip([0, 1], colors, target_names):
    plt.hist(X_lda[y == label], bins=30, alpha=0.6, color=color, label=name)

plt.title("Distribuzione delle classi dopo LDA (1D)")
plt.xlabel("Componente LDA")
plt.ylabel("Frequenza")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
