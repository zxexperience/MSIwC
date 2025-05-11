import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA

#Wczytanie 8 plików z danymi
folder_path = r"C:\Users\PC\Downloads\archive"
datasets = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

dfs = []
column_mismatches = False
reference_columns = None

#Sprawdzenie czy kolumny w nich są takie same
for i, file in enumerate(datasets):
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    if i == 0:
        reference_columns = df.columns
    else:
        if not reference_columns.equals(df.columns):
            column_mismatches = True

    dfs.append(df)
    del df

#Jeżeli są, to można połączyć dane w jeden dataframe
if not column_mismatches:
    combined_df = pd.concat(dfs, ignore_index=True)

    #Zapis połączonych danych do pliku jeżeli będzie potrzebny
    #output_path = os.path.join(folder_path, "combined_dataset.csv")
    #combined_df.to_csv(output_path, index=False)

    #Rozmiar danych do analizy
    rows, cols = combined_df.shape
    print(f'Liczba wierszy: {rows}')
    print(f'Liczba kolumn: {cols}')

    #Usunięcie białych znaków z nazw kolumn
    col_names = {col: col.strip() for col in combined_df.columns}
    combined_df.rename(columns=col_names, inplace=True)

    #Czyszczenie
    len_before = len(combined_df)
    #Kasowanie tych samych wierszy
    duplicates = combined_df[combined_df.duplicated()]
    combined_df.drop_duplicates(inplace=True)

    #Zamiana INF na Na
    combined_df.replace([np.inf, -np.inf, "Infinity"], pd.NA, inplace=True)

    #Zamiana '' na Na
    combined_df.replace('', pd.NA, inplace=True)

    # Konwersja kolumn Flow Bytes/s i Flow Packets/s na liczbowe
    numeric_candidate_cols = ["Flow Bytes/s", "Flow Packets/s"]
    for col in numeric_candidate_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    #Kasowanie pustych wartości
    combined_df.dropna(inplace=True)

    len_after = len(combined_df)
    removed_rows = len_before - len_after
    removed_percent = (removed_rows / len_before) * 100

    print(f"Liczba usuniętych wierszy: {removed_rows} ({removed_percent:.2f}%)")

    #Kasowanie kolumn z tymi samymi wartościami. Zawierają same zera
    num_unique = combined_df.nunique()
    one_variable = num_unique[num_unique == 1]
    not_one_variable = num_unique[num_unique > 1].index

    dropped_cols = one_variable.index
    combined_df = combined_df[not_one_variable]

    print('Skasowane kolumny z tymi samymi wartościami:',dropped_cols)

    # Oddzielenie tabeli
    X = combined_df.drop(columns=["Label"])
    y = combined_df["Label"]

    #Normalizacja wartości pomiędzy 0-1. Algorytm mógłby skupiać się za mocno na dużych liczbach
    scaler = MinMaxScaler()
    X_scaled_numpy = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_numpy, columns=X.columns)

    #Sprawdzenie czy wszystkie kolumny są numeryczne czy wymagają encodingu
    are_all_numeric = X_scaled.dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt)).all()
    print("Wszystkie kolumny są numeryczne:", are_all_numeric)
    print(X_scaled.info())
    #TODO: Tabela Y będzie wymagać zmapowania typów ataków na wartości liczbowe. Z grupowaniem czy bez ?

    #Ekstracja cech
    size = len(X.columns) // 2
    ipca = IncrementalPCA(n_components=size, batch_size=500)
    for batch in np.array_split(X.scaled, len(X) // 500):
        ipca.partial_fit(batch)

    print(f'Zachowane informacje: {sum(ipca.explained_variance_ratio_):.2%}')

    extracted_features_X = ipca.transform(X_scaled)
    new_X = pd.DataFrame(extracted_features_X, columns=[f'PC{i + 1}' for i in range(size)])

    #Dataframe new_X zawiera przetworzone dane dla modelu




