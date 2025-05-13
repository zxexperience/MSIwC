import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt

# Wczytanie danych
X = pd.read_csv("X_clf.csv")
y = pd.read_csv("y_clf.csv").squeeze()

# Check klas
print("Liczność klas w y_clf:")
print(y.value_counts().sort_index())

# Podział na train/test ze stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# One-hot encoding
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# Obliczanie wag klas
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("\nWagi klas:", class_weight_dict)

# Budowa modelu MLP
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Trening modelu
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=512,
    validation_split=0.1,
    class_weight=class_weight_dict,
    verbose=1
)

# Ewaluacja
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nDokładność na zbiorze testowym: {accuracy:.4f}")

# Predykcje
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = y_test.values

# Raport
print("\nUnikalne klasy w y_test:")
print(np.unique(y_true_labels, return_counts=True))

print("\nRaport klasyfikacji:")
print(classification_report(y_true_labels, y_pred_labels, digits=4))

print("Macierz pomyłek:")
print(confusion_matrix(y_true_labels, y_pred_labels))

# Wykres: dokładność
plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Dokładność modelu")
plt.xlabel("Epoka")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_accuracy.png")
plt.close()

# Wykres: strata
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Strata modelu")
plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.close()

print("\nWykresy zapisane jako 'training_accuracy.png' i 'training_loss.png'")
