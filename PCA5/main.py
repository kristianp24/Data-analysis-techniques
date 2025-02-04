import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cerinta1():
    data = pd.read_csv('dataIN/Mortalitate.csv', index_col='Tara')
    c1 = data[data['RS'] < 0]
    c1.to_csv('dataOUT/cerinta1.csv')

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col.mean()])
    return data
def pca_analysys():
    data = pd.read_csv('dataIN/Mortalitate.csv', index_col='Tara')
    cleaned_data = clean_data(data.iloc[:, 1:])

    # Standartizare
    scaler = StandardScaler()
    nromalized_data = scaler.fit_transform(cleaned_data)

    # PCA
    modelPCA = PCA()
    modelPCA.fit(nromalized_data)

    # Variantele
    variances = modelPCA.explained_variance_
    print('Variantele: ', variances)

    # Scorurile
    scoruri = modelPCA.transform(nromalized_data)
    pd.DataFrame(scoruri, columns=['C'+str(i) for i in range(len(variances))]).to_csv('dataOUT/scoruri.csv', index=False)


    # Graficul
    plt.scatter(scoruri[:,0], scoruri[:,1], color='r')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    # Coef de corr
    corr_coef = np.corrcoef(nromalized_data.T, scoruri.T)[:scoruri.shape[1], scoruri.shape[1]:]
    corr_coef_df = pd.DataFrame(corr_coef, columns=['C'+str(i) for i in range(len(variances))], index=cleaned_data.columns)
    print(corr_coef_df)

    # Communalities
    comm = np.cumsum(corr_coef ** 2, axis=1)
    print(comm)


pca_analysys()