import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
def cerinta1(date:pd.DataFrame):
    summed_cols = ['2000', '2005', '2010', '2015', '2018']
    mediile = date.apply(func=lambda x: x[summed_cols].mean(), axis=1)
    mediile.name = 'Media'
    cer1 = pd.concat([date['Code'], mediile], axis=1)
    cer1.to_csv('dataOUT/cerinta1.csv', index=False)

def cerinta2(date:pd.DataFrame):
    summed_cols = ['2000', '2005', '2010', '2015', '2018']
    aux = date.groupby('Continent').mean(summed_cols).reset_index()
    years = aux.iloc[:,1:].apply(func= lambda x: x.idxmax(), axis = 1)
    years.name = 'Anul'
    cer2 = pd.concat([aux['Continent'], years], axis=1)
    cer2.to_csv('dataOUT/cerinta2.csv', index=False)

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data

def cluster_analysys(alcohol:pd.DataFrame):
    numeric_data = alcohol.iloc[:,2:]
    cleaned_data = clean_data(numeric_data)

    # Standartizam datele
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cleaned_data)

    # Matricea de irearhie
    linkage_matrix = linkage(normalized_data, method='ward')
    columns_linkage = ['Cluster1', 'Cluster2', 'Distances', 'No. Clusters']
    df_linkage = pd.DataFrame(linkage_matrix, columns=columns_linkage)
    df_linkage.to_csv('dataOUT/matricea_ierarhie.csv')

    #Nr optim cluster, metoda elbow
    # distances = linkage_matrix[:, 2]
    # values = linkage_matrix[:, 3]
    # dif  = np.diff(distances, 2)
    # index_max = np.argmax(dif) + 1
    # value = values[index_max]
    # print('Nr optim de cluster dupa metoda elbow: ', index_max)

    # Nr optim cluster, silouhette score
    nr_cluster_optim = 0
    best_score = 0
    scores = []

    for nr in range(2,10):
        labels = fcluster(linkage_matrix, nr, criterion='maxclust')
        score = silhouette_score(normalized_data, labels)
        if score > best_score:
            best_score = score
            nr_cluster_optim = nr
        scores.append(score)
    print("Best silohette score: ", best_score)
    print("Numar optim de clustere: ", nr_cluster_optim)

    plt.plot([i for i in range(2,10)], scores)
    plt.xlabel('Nr clustere')
    plt.ylabel('Scores')
    plt.show()

    dendrogram(linkage_matrix, no_labels=nr_cluster_optim)
    plt.show()

    # partitia optima
    labels_optim = fcluster(linkage_matrix, nr_cluster_optim, criterion='maxclust')
    alcohol['Clust ID'] = labels_optim
    print(alcohol)


alcohol = pd.read_csv('DateIN/alcohol.csv')
alcohol_countryindex =pd.read_csv('DateIN/alcohol.csv', index_col=0)
contintente = pd.read_csv('DateIN/CoduriTariExtins.csv')
merged_data = alcohol.merge(contintente, left_on='Entity', right_on='Tari', how='inner')
cluster_analysys(alcohol)
