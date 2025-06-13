import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def main():
    # --- 1. Definizione degli Insiemi e delle Mappe ---
    states = ["Fermo", "Camminare", "Correre"]
    n_states = len(states)

    observations_labels = ["Quiet", "Soft", "Medium", "Loud"]
    n_observations = len(observations_labels)

    obs_label_to_idx = {label: i for i, label in enumerate(observations_labels)}
    idx_to_obs_label = {i: label for i, label in enumerate(observations_labels)}
    idx_to_state_label = {i: label for i, label in enumerate(states)}
    state_label_to_idx = {label: i for i, label in enumerate(states)}


    # --- 2. Definizione delle Matrici di Probabilità HMM ---
    start_prob = np.array([0.5, 0.3, 0.2])

    trans_mat = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.7, 0.2],
        [0.05, 0.25, 0.7]
    ])

    emission_mat = np.array([
        [0.7,   0.2,   0.08,  0.02],
        [0.1,   0.6,   0.25,  0.05],
        [0.02,  0.08,  0.3,   0.6]
    ])


    # --- 3. Inizializzazione e Assegnazione Parametri del Modello HMM ---
    model = hmm.MultinomialHMM(n_components=n_states, n_trials=1, init_params="", random_state=None)

    model.startprob_ = start_prob
    model.transmat_ = trans_mat
    model.emissionprob_ = emission_mat


    # --- 4. Generazione di una Sequenza Casuale di Stati e Osservazioni ---
    sequence_length = 150

    true_observations_one_hot, true_state_indices = model.sample(sequence_length)

    true_observed_sequence_int = np.argmax(true_observations_one_hot, axis=1)
    true_observed_sequence_labels = [idx_to_obs_label[idx] for idx in true_observed_sequence_int]
    true_state_labels = [idx_to_state_label[idx] for idx in true_state_indices]


    # --- 5. Decodifica della Sequenza di Osservazioni Generata ---
    logprob, predicted_state_indices = model.decode(true_observations_one_hot,
                                                    lengths=[sequence_length],
                                                    algorithm="viterbi")

    predicted_state_labels = [idx_to_state_label[s] for s in predicted_state_indices]


    # --- 6. Calcolo delle Metriche di Valutazione ---
    accuracy = accuracy_score(true_state_labels, predicted_state_labels)
    conf_matrix = confusion_matrix(true_state_labels, predicted_state_labels, labels=states)


    # --- 7. Plot in un'Unica Finestra (A.1, A.2) ---
    # Creiamo una figura con 4 subplot: Sequenza, Matrice Confusione, Matrice Transizione, Matrice Emissione
    fig, axs = plt.subplots(4, 1, figsize=(18, 26)) # Aumentate ancora le dimensioni per maggiore leggibilità

    state_colors = {
        "Fermo": "lightgreen",
        "Camminare": "orange",
        "Correre": "red"
    }

    # Plot 1: Sequenza di Osservazioni con Stati Predetti e Reali
    axs[0].set_title(f"Sequenza di Osservazioni con Stati Reali e Predetti (Accuratezza: {accuracy*100:.2f}%)")
    time_points = np.arange(sequence_length)

    # Plotta le osservazioni come punti
    axs[0].plot(time_points, true_observed_sequence_int, 'o-', color='blue',
                label='Osservazione Generata', markersize=4, linewidth=1, alpha=0.7)

    # Plot delle sequenze di stati (reali e predetti)
    true_state_plot_values = [state_label_to_idx[s] for s in true_state_labels]
    predicted_state_plot_values = [state_label_to_idx[s] for s in predicted_state_labels]

    axs[0].plot(time_points, true_state_plot_values, 'k--', label='Stato Reale', linewidth=1.5, alpha=0.8) # Linea nera tratteggiata
    axs[0].plot(time_points, predicted_state_plot_values, 'r-', label='Stato Predetto', linewidth=1.5, alpha=0.6) # Linea rossa continua

    min_obs_val = min(obs_label_to_idx.values())
    max_obs_val = max(obs_label_to_idx.values())
    plot_range = max_obs_val - min_obs_val + 1

    for i, state_label in enumerate(predicted_state_labels):
        axs[0].add_patch(Rectangle(
            (i - 0.5, min_obs_val - 0.5),
            1,
            plot_range,
            facecolor=state_colors[state_label],
            alpha=0.1,
            edgecolor='none'
        ))

    # Imposta le etichette dell'asse Y solo per le osservazioni
    axs[0].set_yticks(list(obs_label_to_idx.values()))
    axs[0].set_yticklabels(observations_labels)
    axs[0].set_ylim(min_obs_val - 0.5, max_obs_val + 0.5) # Limita l'asse Y alle osservazioni

    # Aggiungi linee orizzontali tratteggiate per gli stati per distinguerli visivamente
    for i, state_name in enumerate(states):
        axs[0].axhline(y=state_label_to_idx[state_name], color='gray', linestyle=':', linewidth=0.8, alpha=0.7,
                       label=f'Livello Stato {state_name}' if i == 0 else "") # Aggiungi etichetta solo una volta

    # Legenda più pulita e in alto a destra
    # Unisci le legende per le linee, gli sfondi e le linee orizzontali
    line_handles, line_labels = axs[0].get_legend_handles_labels()
    bg_handles = [Rectangle((0,0),1,1, fc=state_colors[s], alpha=0.1) for s in states] # Sfondo Stati
    full_handles = line_handles + bg_handles
    full_labels = line_labels + [f'Sfondo Stato {s}' for s in states] # Etichette per gli sfondi
    axs[0].legend(full_handles, full_labels, title="Legenda", loc='upper right', ncol=2, fontsize='small')

    axs[0].set_xlabel("Tempo")
    axs[0].set_ylabel("Osservazione (Valore)")
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)


    # Plot 2: Matrice di Confusione
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
                xticklabels=states, yticklabels=states, ax=axs[1],
                annot_kws={"size": 12}) # Aumenta la dimensione del font dei numeri
    axs[1].set_title("Matrice di Confusione per la Decodifica degli Stati")
    axs[1].set_xlabel("Stato Predetto")
    axs[1].set_ylabel("Stato Reale")
    axs[1].tick_params(axis='x', labelsize=10) # Riduci la dimensione del font delle etichette
    axs[1].tick_params(axis='y', labelsize=10)


    # Plot 3: Matrice di Transizione del Modello
    sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=states, yticklabels=states, ax=axs[2],
                annot_kws={"size": 12}) # Aumenta la dimensione del font dei numeri
    axs[2].set_title("Matrice di Transizione del Modello (Impostata)")
    axs[2].set_xlabel("Stato Successivo")
    axs[2].set_ylabel("Stato Corrente")
    axs[2].tick_params(axis='x', labelsize=10)
    axs[2].tick_params(axis='y', labelsize=10)


    # Plot 4: Matrice di Emissione del Modello
    sns.heatmap(model.emissionprob_, annot=True, fmt=".2f", cmap='Purples',
                xticklabels=observations_labels, yticklabels=states, ax=axs[3],
                annot_kws={"size": 12}) # Aumenta la dimensione del font dei numeri
    axs[3].set_title("Matrice di Emissione del Modello (Impostata)")
    axs[3].set_xlabel("Osservazione")
    axs[3].set_ylabel("Stato")
    axs[3].tick_params(axis='x', labelsize=10)
    axs[3].tick_params(axis='y', labelsize=10)


    plt.tight_layout() # Assicura che i subplot non si sovrappongano
    plt.show()

    # --- 8. Stampa delle Metriche a Console ---
    print("\n" + "="*50)
    print("HMM per Attività Motoria - Riepilogo")
    print("="*50)
    print(f"\nProbabilità Logaritmica della Sequenza Decodificata: {logprob:.2f}")
    print(f"\nAccuratezza della Decodifica: {accuracy*100:.2f}%")
    print("\nMatrice di Confusione:\n", conf_matrix)

if __name__ == "__main__":
    main()