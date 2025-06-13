import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 1. Carica dati di esempio (2 feature per la visualizzazione 2D)
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6)

# 2. Crea ed allena una SVM lineare
clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# 3. Funzione per disegnare iperpiano e margini
def plot_svm(clf, X, y):
    plt.figure(figsize=(8, 6))
    
    # Plot dei punti
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30)
    
    # Plot dei vettori di supporto
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    # Crea griglia
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot dell'iperpiano e margini
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               alpha=0.7, linestyles=['--', '-', '--'])

    plt.title("Support Vector Machine (SVM) con margini")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# 4. Mostra il grafico
plot_svm(clf, X, y)


