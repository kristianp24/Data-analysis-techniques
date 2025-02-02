import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import  StandardScaler
import seaborn as sb

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data
def pca_analysys():
    data = pd.read_csv('dataIN/MortalitateEU.csv')
    numeric_data = data.iloc[:, 2:]
    cleaned_data = clean_data(numeric_data)

    # Standartizam datele
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cleaned_data)

    # Performam PCA
    pca_Model = PCA()
    X = pca_Model.fit(normalized_data)

    scoruri = pca_Model.transform(normalized_data)
    scoruri_df = pd.DataFrame(scoruri, columns=['PC'+str(i+1) for i in range(scoruri.shape[1])])
    scoruri_df.to_csv('dataOUT/scoruri.csv')

    # varianta componente
    varianta = pca_Model.explained_variance_ratio_
    print(varianta)
    # Plot varianta componente
    plt.scatter(scoruri_df['PC1'], scoruri_df['PC2'])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    #Corelatii factoriale
    corr = np.corrcoef(normalized_data.T, scoruri.T)[:len(numeric_data.columns), len(numeric_data.columns):]
    df_corr = pd.DataFrame(corr, index=numeric_data.columns,columns=['PC'+str(i+1) for i in range(scoruri.shape[1])])
    df_corr.to_csv('dataOUT/correlatii.csv')
    # Plot corelatii
    sb.heatmap(df_corr)
    plt.show()

    #Comunalitatii
    comm = np.cumsum(corr ** 2, axis=1)
    df_comm = pd.DataFrame(comm, index=numeric_data.columns,columns=['PC'+str(i+1) for i in range(scoruri.shape[1])])
    df_comm.to_csv('dataOUT/Comunalitatii.csv')
    # corelograma comunalitai
    sb.heatmap(df_comm)
    plt.show()
pca_analysys()