import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Wczytanie danych BENIGN (do nauki autoencodera)
X_train = pd.read_csv("X_ae.csv")

# Wczytanie pełnego zbioru (do testowania detekcji anomalii)
X_all = pd.read_csv("X_clf.csv")

# Normalizacja (jeśli nie była wcześniej zrobiona)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_all_scaled = scaler.transform(X_all)

# Architektura autoencodera
input_dim = X_train_scaled.shape[1]
encoding_dim = input_dim // 2  # kompresja 50%

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Trening autoencodera
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=512,
    shuffle=True,
    callbacks=[early_stop],
    verbose=1
)

# Rekonstrukcja danych testowych
X_pred = autoencoder.predict(X_all_scaled)
reconstruction_errors = np.mean(np.square(X_all_scaled - X_pred), axis=1)

# Automatyczny próg wykrywania anomalii
threshold = np.percentile(reconstruction_errors, 90)

# Wykryte anomalie
anomaly_flags = reconstruction_errors > threshold
n_anomalies = np.sum(anomaly_flags)

print(f"\nPróg detekcji anomalii (90 percentyl): {threshold:.6f}")
print(f"Wykryto {n_anomalies} potencjalnych anomalii na {len(X_all)} przykładów.")

# Wykres błędów rekonstrukcji
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_errors, bins=100, alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label='Próg anomalii')
plt.title("Histogram błędów rekonstrukcji")
plt.xlabel("Błąd rekonstrukcji")
plt.ylabel("Liczba przypadków")
plt.legend()
plt.tight_layout()
plt.savefig("reconstruction_error_hist.png")
plt.close()

print("Zapisano wykres do 'reconstruction_error_hist.png'")
y_true = pd.read_csv("y_clf.csv").squeeze()
n_true_attacks = y_true[anomaly_flags].value_counts()
print("\nWśród wykrytych anomalii:")
print(n_true_attacks)


