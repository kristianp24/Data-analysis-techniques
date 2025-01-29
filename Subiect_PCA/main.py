import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb

def cerinta1():
    netflix = pd.read_csv('dateIN/Netflix.csv')
    netflix.fillna(0, inplace=True)
    numeric_data = netflix.iloc[:, 2:]
    scaler = StandardScaler()
    standardData = scaler.fit_transform(numeric_data)
    netflix.iloc[:,2:] = standardData
    netflix.sort_values(by='Internet', ascending=False, inplace=True)
    netflix.to_csv('dateIN/cerinta1.csv', index=False)

def cerinta2():
    netflix = pd.read_csv('dateIN/Netflix.csv')
    codTari = pd.read_csv('dateIN/CoduriTari.csv')
    mean = netflix.iloc[:, 2:].mean()
    netflix.fillna(mean, inplace=True)
    merged_data = netflix.merge(codTari, on='Cod', how='left')
    merged_data.fillna(mean, inplace=True)
    print(merged_data.columns)
    columns = netflix.columns[2:10]
    grouped = merged_data.groupby('Continent')[columns].std()
    grouped2 = merged_data.groupby('Continent')[columns].mean()
    coef_VAR = grouped / grouped2
    cer2 = pd.DataFrame(coef_VAR)
    cer2.fillna(0, inplace=True)
    cer2.sort_values(by='Librarie', ascending=False, inplace=True)
    cer2.to_csv('dateOUT/cerinta2.csv', index=True)

def pca_():
    #citim datele
    netflix = pd.read_csv('dateIN/Netflix.csv')
    codTari = pd.read_csv('dateIN/CoduriTari.csv')

    # curatam setul de date
    mean = netflix.iloc[:, 2:].mean()
    netflix.fillna(mean, inplace=True)

    # facem merge intre date
    merged_data = netflix.merge(
        codTari,
        on='Cod'
    )
    print(merged_data.head())

    # standartizam datele numerice
    numeric_data = merged_data.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numeric_data)

    # facem PCA
    acp = PCA()
    # acestia sunt si scorurile
    scoruri_acp = acp.fit_transform(normalized_data)

    # varianta componentelor principale
    variance = acp.explained_variance_ratio_
    print(variance)

    # dataframe cu scorurile
    scoruri_df = pd.DataFrame(scoruri_acp, columns=['PC'+str(i+1) for i in range(len(variance))])
    print(scoruri_df)
    scoruri_df.to_csv('dateOUT/scores.csv')

    # Vizualizare componente principale
    plt.figure(figsize=(10,7))
    plt.scatter(scoruri_df['PC1'], scoruri_df['PC2'])
    plt.show()

    # Corelatii factoriale
    cor_fact = np.corrcoef(normalized_data.T, scoruri_acp.T)[:len(numeric_data.columns), len(numeric_data.columns):]
    sb.heatmap(cor_fact)
    plt.show()

    # comunalitati
    com = np.cumsum(cor_fact ** 2, axis=1)
    comunalitati_df = pd.DataFrame(
        com,
        index=numeric_data.columns,
        columns=['PC'+str(i+1) for i in range(len(variance))]
    )
    print(comunalitati_df.head())
    comunalitati_df.to_csv('dateOUT/comunalitati.csv')
    sb.heatmap(comunalitati_df)
    plt.show()





pca_()