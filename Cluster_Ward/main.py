import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def ceritna1():
    alcool = pd.read_csv('dataIN/alcohol.csv')
    mediile = pd.Series(alcool.apply(func=lambda x: x[2:].mean(), axis=1))
    mediile.name = 'Consum_Mediu'

    data = {'Code': alcool['Code'],
            'Country':alcool['Country'],
            'Consum_Medie' :mediile
            }
    cer1 = pd.DataFrame(data).sort_values(by='Consum_Medie', ascending=False)
    cer1.to_csv('dataOUT/cerinta1.csv')

def cerinta2():
    alcool = pd.read_csv('dataIN/alcohol.csv')
    coduri = pd.read_csv('dataIN/CoduriTariExtins.csv')
    merged_data = alcool.merge(coduri, on='Country')

    grouped_data = merged_data.groupby(by='Continent')[alcool.columns[2:]].mean()
    max_years = grouped_data.apply(func=lambda  x: x.idxmax(), axis=1)
    cer2 = pd.DataFrame(max_years, columns=['Anul'])
    cer2.to_csv('dataOUT/cerinta2.csv', index=True)

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data
def cluster_analysys():
    alcool = pd.read_csv('dataIN/alcohol.csv')
    cleaned_data = clean_data(alcool.iloc[:, 2:])

    # Normalizare
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cleaned_data)

    # linkage matrix
    linkage_matrix = linkage(normalized_data, method='ward')
    linkage_matrix_df = pd.DataFrame(linkage_matrix, columns=['Cluster1', 'Cluster2', 'Distanta', 'Nr_clusteri'])
    linkage_matrix_df.to_csv('dataOUT/matricea_ierarhie.csv')
    print(linkage_matrix_df.head())

    # Componenta cu 5 clusteri
    nr_clusters = 5
    labels = fcluster(linkage_matrix, nr_clusters, criterion='maxclust')
    alcool['ClustID'] = labels
    alcool.drop(alcool.columns[2:-1], inplace=True, axis=1)
    alcool.to_csv('dataOUT/p4.csv')

    # Plot 5 clusteri
    modelPCA = PCA()
    S = modelPCA.fit_transform(normalized_data)

    linkage_matrix_2 = linkage(S, method='ward')
    labels2 = fcluster(linkage_matrix_2, nr_clusters, criterion='maxclust')

    for cluster in np.unique(labels2):
        plt.scatter(S[labels2 == cluster, 0], S[labels2 == cluster, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


cluster_analysys()