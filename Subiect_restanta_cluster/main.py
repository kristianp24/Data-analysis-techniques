import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage


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


def linkage_matrix(date:pd.DataFrame):
    date = date.fillna(0)
    linkage_matrix = linkage(date, method='ward')
    columns = ['Cluster1', 'Cluster2', 'Distanta', 'No.clusters']
    df = pd.DataFrame(linkage_matrix, columns=columns)
    print(df)
    return linkage_matrix

def partitia_optimala(linkage_data, date_original:pd.DataFrame, indexi):
    # gasim nr optimal de clusteri prin silouhette score
    best_score = 0
    nr_cluster_optim = 1
    scores = []
    for nr in range(2,10):
        labels = fcluster(linkage_data, nr, 'maxclust')
        score = silhouette_score(date_original, labels=labels)
        scores.append(score)
        if (score > best_score):
            best_score = score
            nr_cluster_optim = nr

    print('Best silouhette score:', score)
    print('Nr optim de cluster:', nr_cluster_optim)

    plt.figure(figsize=(10,7))
    dendrogram(linkage_data, labels=indexi)
    plt.show()

def componenta_partitiei_optime(data_original:pd.DataFrame, linkage_matrix):
    labels = fcluster(linkage_matrix, 3, 'maxclust')
    labels1 = pd.Series(labels, name='Cluster ID')
    df = pd.concat([data_original, labels1], axis=1)
    df.to_csv('dataOUT/popt.csv', index=False)



indicatori = pd.read_csv('dataIN/Indicatori.csv')
populatie = pd.read_csv('dataIN/PopulatieLocalitati.csv')
locationaQ = pd.read_csv('dataIN/LocationQ.csv')
date_cluster = locationaQ.iloc[:, 1:]
print(date_cluster)
# linkage_data = linkage_matrix(date_cluster)
# # print([locationaQ['Judet']])
# partitia_optimala(linkage_data, date_cluster, locationaQ['Judet'].tolist())
# componenta_partitiei_optime(locationaQ, linkage_data)
cerinta2(indicatori, populatie)


