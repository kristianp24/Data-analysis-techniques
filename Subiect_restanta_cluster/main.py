import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from kneed import KneeLocator


def cerinta1(indicatori:pd.DataFrame):
    no_of_cities = indicatori.shape[0]
    avg_CFA = indicatori['CFA'].mean()
    df = indicatori.apply(lambda x: x if x['CFA'] > avg_CFA else 0, axis=1)
    cer1 = pd.DataFrame(df).sort_values(ascending=False,by='CFA')
    cer1.to_csv('dataOUT/cerinta1.csv', index=False)


def cerinta2(indicatori:pd.DataFrame, populatie:pd.DataFrame):
    df = indicatori.merge(populatie, left_index=True, right_index=True)
    columns_for_sum = ['NR_FIRME', 'NSAL', 'CFA', 'PROFITN', 'PIERDEREN', 'Populatie']
    aux = pd.DataFrame(df.groupby('Judet')[columns_for_sum].sum())

    # aux = aux.merge(populatie, left_index=True, right_index=True)
    for col in ['NR_FIRME', 'NSAL', 'CFA', 'PROFITN', 'PIERDEREN']:
        aux[col] = aux[col] * 1000 / aux['Populatie']

    aux.to_csv('dataOUT/cerinta2.csv', index=False)
    print(aux)


def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())

    return data

def cluster_analysys(locationaQ:pd.DataFrame):
    numeric_data = locationaQ.iloc[:,1:]
    cleaned_data = clean_data(numeric_data)

    # Standartizam data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cleaned_data)

    # Matricea de ierarhie
    matrix_I = linkage(normalized_data, method='ward')
    columns = ['Cluster1', 'Cluster2', 'Distanta', 'Nr.clustere']
    df_matrice_ierarhie = pd.DataFrame(matrix_I, columns=columns)
    print(df_matrice_ierarhie)

    # Dendrograma
    dendrogram(matrix_I)
    plt.show()

    # Elbow method pentru partitia optimala
    distances = matrix_I[:, 2]
    diff = np.diff(distances, 2)
    elbow_index = np.argmax(diff) + 1
    values = matrix_I[:, 3]
    elbow_value = values[elbow_index]
    print('Numarul optim de clusteri dupa punctul elbow: ', elbow_value)

    # Silouhette score la nivel de partitie
    nr_cluster_optim = 1
    best_score = 0
    scores = []

    for nr in range(2,10):
        labels = fcluster(matrix_I, nr, criterion='maxclust')
        score = silhouette_score(normalized_data, labels)
        if score > best_score:
            best_score = score
            nr_cluster_optim  = nr
        scores.append(score)
    print('Cel mai bun scor: ', best_score)
    print('Nr de clustere: ', nr_cluster_optim)

    # Ploturi
    plt.plot([i for i in range(2,10)], scores)
    plt.xlabel('Nr clusters')
    plt.ylabel('Silouhette score')
    plt.show()

    # Numarul optim de clustere = 3
    # Partitia optima
    labels_optim = fcluster(matrix_I, nr_cluster_optim, criterion='maxclust')
    locationaQ['ClusterID'] = labels_optim
    print(locationaQ)

    #Silouhette score la nivel de instanta
    silh_score_instanta = silhouette_samples(normalized_data, labels_optim)
    print(silh_score_instanta)


indicatori = pd.read_csv('dataIN/Indicatori.csv')
populatie = pd.read_csv('dataIN/PopulatieLocalitati.csv')
locationaQ = pd.read_csv('dataIN/LocationQ.csv')
date_cluster = locationaQ.iloc[:, 1:]
print(date_cluster)
# linkage_data = linkage_matrix(date_cluster)
# # print([locationaQ['Judet']])
# partitia_optimala(linkage_data, date_cluster, locationaQ['Judet'].tolist())
# componenta_partitiei_optime(locationaQ, linkage_data)
cluster_analysys(locationaQ)


