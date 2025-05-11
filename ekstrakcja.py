import pandas as pd
import numpy as np
import os

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
    print(f'Number of rows: {rows}')
    print(f'Number of columns: {cols}')

    #Usunięicie białych znaków z nazw kolumn
    col_names = {col: col.strip() for col in combined_df.columns}
    combined_df.rename(columns=col_names, inplace=True)

    #Czyszczenie
    len_before = len(combined_df)
    #Kasowanie tych samych wierszy
    duplicates = combined_df[combined_df.duplicated()]
    combined_df.drop_duplicates(inplace=True)

    #Zamiana INF na Na
    combined_df.replace([np.inf, -np.inf], pd.NA, inplace=True)

    #Zamiana '' na Na
    combined_df.replace('', pd.NA, inplace=True)

    #Kasowanie pustych wartości
    combined_df.dropna(inplace=True)

    len_after = len(combined_df)
    removed_rows = len_before - len_after
    removed_percent = (removed_rows / len_before) * 100

    print(f"Liczba usuniętych wierszy: {removed_rows} ({removed_percent:.2f}%)")

    #Kasowanie kolumn z tymi samymyi wartościami
    num_unique = combined_df.nunique()
    one_variable = num_unique[num_unique == 1]
    not_one_variable = num_unique[num_unique > 1].index

    dropped_cols = one_variable.index
    combined_df = combined_df[not_one_variable]

    print('Dropped columns:',dropped_cols)

