from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def clen_data(numeric_data:pd.DataFrame):
    if numeric_data.isna().any().any():
        for col in numeric_data.columns:
            if numeric_data[col].isna().any():
                numeric_data[col] = numeric_data[col].fillna(numeric_data[col].mean())

    return numeric_data


def pca_analysys():
    data = pd.read_csv('dataIN/MiseNatPopTari.csv')
    numeric_data = data.select_dtypes(['float64', 'int64'])
    data_to_standartize = clen_data(numeric_data.iloc[:, 1:])
    print(data_to_standartize)
    # Standartizam data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_to_standartize)

    # Facem PCA
    pcaModel = PCA()
    scores = pcaModel.fit_transform(normalized_data)
    # data Frame cu scoruri
    pca_components = ['PC'+str(i+1) for i in range(scores.shape[1])]
    df_scores = pd.DataFrame(scores, columns=pca_components)
    df_scores.to_csv('dataOUT/scores.csv')

    # varianta
    variance = pcaModel.explained_variance_ratio_

    #Vizualzare componente Principale
    # plt.figure(figsize=(10,7))
    # plt.scatter(df_scores['PC1'], df_scores['PC2'])
    # plt.show()

    # Corelatii factoriale -> cat de bine este reprez o variabila intiala de catre fiecare factor
    cor_fact = np.corrcoef(normalized_data.T, scores.T)[:len(data_to_standartize.columns), len(data_to_standartize.columns):]
    sb.heatmap(cor_fact)
    plt.show()
    print(variance)

    #Comunalitati
    com = np.cumsum(cor_fact**2, axis=1)
    com_df = pd.DataFrame(
        com,
        index=data_to_standartize.columns,
        columns=pca_components

    )
    com_df.to_csv('dataOUT/comunalitatii.csv')




pca_analysys()