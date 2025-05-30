import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA

#Wczytanie plików z danymi
folder_path = r"archive"
datasets = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

dfs = []
column_mismatches = False
reference_columns = None

#Sprawdzenie czy kolumny w nich są takie same
for i, file in enumerate(datasets):
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if i == 0:
        reference_columns = df.columns
    else:
        if not reference_columns.equals(df.columns):
            raise ValueError(f"Column mismatch for file {file}")

    dfs.append(df)
    del df

#Jeżeli są, to można połączyć dane w jeden dataframe
combined_df = pd.concat(dfs, ignore_index=True)

print("\nDostępne etykiety w danych:")
print(combined_df["Label"].value_counts())

#Usunięcie białych znaków z nazw kolumn
col_names = {col: col.strip() for col in combined_df.columns}
combined_df.rename(columns=col_names, inplace=True)

#Czyszczenie
len_before = len(combined_df)
combined_df.drop_duplicates(inplace=True)
combined_df.replace([np.inf, -np.inf, "Infinity", ''], pd.NA, inplace=True)

numeric_cols = ["Flow Bytes/s", "Flow Packets/s"]
for col in numeric_cols:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

combined_df.dropna(inplace=True)

len_after = len(combined_df)
print(f"\nUsunięto {len_before - len_after} wierszy z brakującymi danymi ({(len_before - len_after) / len_before:.2%})")

#Usunięcie kolumn o jednej wartości
num_unique = combined_df.nunique()
one_variable = num_unique[num_unique == 1]
combined_df = combined_df[num_unique[num_unique > 1].index]

if 'Destination Port' in combined_df.columns:
    combined_df.drop(columns=['Destination Port'], inplace=True)

print("\nKolumny usunięte (stała wartość):", one_variable.index.tolist())

# Filtrowanie etykiet
attack_labels = ["DoS Hulk", "DDoS"]
combined_df = combined_df[combined_df["Label"].isin(["BENIGN"] + attack_labels)].copy()

# Oddzielenie cech od etykiet
X = combined_df.drop(columns=["Label"])
y = combined_df["Label"]

# Sprawdzenie typu danych
are_all_numeric = X.dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt)).all()
print("\nWszystkie kolumny są numeryczne:", are_all_numeric)

# Normalizacja
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(f"X: {X_scaled.columns.tolist()}")

# Ekstrakcja cech
size = len(X.columns) // 2
ipca = IncrementalPCA(n_components=size, batch_size=500)
for batch in np.array_split(X_scaled, len(X_scaled) // 500 or 1):
    ipca.partial_fit(batch)

print(f"\nZachowane informacje po PCA: {sum(ipca.explained_variance_ratio_):.2%}")

X_pca = pd.DataFrame(ipca.transform(X_scaled), columns=[f'PC{i+1}' for i in range(size)])

y = y.reset_index(drop=True)

# Mapowanie etykiet
label_map = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "DDoS": 2
}
y_clf = y.map(label_map)

# Dane do klasyfikatora
X_clf = X_pca.copy()

# Dane do autoencodera – tylko BENIGN
X_ae = X_pca[y_clf == 0].reset_index(drop=True)

print("\nGotowe dane:")
print(f"- X_clf (dla klasyfikatora): {X_clf.shape}")
print(f"- y_clf (etykiety): {y_clf.value_counts().sort_index().to_dict()}")
print(f"- X_ae (dla autoencodera): {X_ae.shape}")

# Zapis danych
os.makedirs("data", exist_ok=True)
X_clf.to_csv("data/X_clf.csv", index=False)
y_clf.to_csv("data/y_clf.csv", index=False)
X_ae.to_csv("data/X_ae.csv", index=False)

print("\nZapisano pliki: X_clf.csv, y_clf.csv, X_ae.csv")
