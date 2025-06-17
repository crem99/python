import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hmmlearn import hmm
from sklearn.utils.validation import check_is_fitted

# Prepare parameters for a 4-components HMM
startprob = np.array([0.6, 0.3, 0.1, 0.0])
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])
means = np.array([[0.0, 0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])
covars = .5 * np.tile(np.identity(2), (4, 1, 1))

# Build an HMM instance and set parameters
gen_model = hmm.GaussianHMM(n_components=4, covariance_type="full")
gen_model.startprob_ = startprob
gen_model.transmat_ = transmat
gen_model.means_ = means
gen_model.covars_ = covars

# Generate samples
X, Z = gen_model.sample(500)

# Plot the sampled data
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
        mfc="orange", alpha=0.7)
for i, m in enumerate(means):
    ax.text(m[0], m[1], 'Component %i' % (i + 1),
            size=17, horizontalalignment='center',
            bbox=dict(alpha=.7, facecolor='w'))
ax.legend(loc='best')
fig.show()

# %%
# Recover the parameters using BIC and AIC to choose the best model
def compute_bic_aic(model, X):
    check_is_fitted(model)
    log_likelihood = model.score(X)
    n_params = (
        model.n_components - 1 +
        model.n_components * (model.n_components - 1) +
        model.n_components * X.shape[1] +
        model.n_components * X.shape[1] * (X.shape[1] + 1) / 2
    )
    bic = -2 * log_likelihood + n_params * np.log(len(X))
    aic = -2 * log_likelihood + 2 * n_params
    return bic, aic

scores = []
models = []
bics = []
aics = []

for n_components in (3, 4, 5, 6):
    for idx in range(5):
        model = hmm.GaussianHMM(n_components=n_components,
                                covariance_type='full',
                                random_state=idx)
        model.fit(X[:X.shape[0] // 2]) # 50/50 train/validate
        val_score = model.score(X[X.shape[0] // 2:])
        scores.append(val_score)
        models.append(model)
        bic, aic = compute_bic_aic(model, X[X.shape[0] // 2:])
        bics.append(bic)
        aics.append(aic)
        print(f'Stati: {n_components}, Random: {idx}, '
              f'Score: {val_score:.2f}, BIC: {bic:.2f}, AIC: {aic:.2f}')

# Seleziona il miglior modello in base al BIC
best_bic_idx = np.argmin(bics)
model = models[best_bic_idx]
n_states = model.n_components
print(f'\n🎯 Il miglior modello ha {n_states} stati (BIC più basso = {bics[best_bic_idx]:.2f})')

# Viterbi decoding
states = model.predict(X)

# %%
# Plot risultati e confronto matrici di transizione
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(X[:, 0], X[:, 1], ".-", label="osservazioni", ms=6, mfc="orange", alpha=0.7)
for i, m in enumerate(means):
    axs[0, 0].text(m[0], m[1], f'Comp {i+1}', size=12, ha='center',
                   bbox=dict(alpha=.7, facecolor='w'))
axs[0, 0].legend()
axs[0, 0].set_title("Punti osservati")

axs[0, 1].plot(Z, label="Generati")
axs[0, 1].plot(states, label="Trovati")
axs[0, 1].set_title("Stati generati vs trovati")
axs[0, 1].set_xlabel("Tempo")
axs[0, 1].set_ylabel("Stato")
axs[0, 1].legend()

sns.heatmap(gen_model.transmat_, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=[f"S{i}" for i in range(gen_model.n_components)],
            yticklabels=[f"S{i}" for i in range(gen_model.n_components)],
            ax=axs[1, 0], cbar=True)
axs[1, 0].set_title("Matrice di transizione generata")
axs[1, 0].set_xlabel("Stato successivo")
axs[1, 0].set_ylabel("Stato attuale")

sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=[f"S{i}" for i in range(model.n_components)],
            yticklabels=[f"S{i}" for i in range(model.n_components)],
            ax=axs[1, 1], cbar=True)
axs[1, 1].set_title("Matrice di transizione trovata")
axs[1, 1].set_xlabel("Stato successivo")
axs[1, 1].set_ylabel("Stato attuale")

plt.tight_layout()
plt.show()
