import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
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

def cluster_Ward(data:pd.DataFrame):
    data = data.fillna(0)
    linkage_data = linkage(data, method='ward')
    columns = ['Cluster1', 'Cluster2', 'Distance', 'No. clusters']
    df = pd.DataFrame(linkage_data, columns=columns)
    print (df)
    return linkage_data

def partitia_optimala_dendograma(linkage_data, original_data, alcohol):
    best_score = 0;
    no_clusters_maxim = 10;
    no_clusters_optim = -1
    scores = []

    for nr in range(2,no_clusters_maxim+1):
        labels = fcluster(linkage_data, nr, 'maxclust')
        silhouette = silhouette_score(original_data, labels)
        scores.append(silhouette)
        if silhouette > best_score:
            best_score = silhouette
            no_clusters_optim = nr

    print('Best silhouette score:', best_score)
    print('No. clusters optim:', no_clusters_optim)


    plt.figure( figsize = (10,7))
    dendrogram(linkage_data, labels= alcohol.index)
    plt.title("Dendrograma")
    plt.xlabel('Tarile')
    plt.ylabel('Distanta')
    plt.grid(True)
    plt.show()

def componenta_partitiei_optimala(linkage_matrix, alcohol_data):
    labels = fcluster(linkage_matrix, 2, 'maxclust')
    series_labels = pd.Series(labels, name='Cluster ID')
    df = pd.concat((alcohol_data, series_labels), axis=1)
    df.to_csv('dataOUT/partiti_optime.csv')






alcohol = pd.read_csv('DateIN/alcohol.csv')
alcohol_countryindex =pd.read_csv('DateIN/alcohol.csv', index_col=0)
contintente = pd.read_csv('DateIN/CoduriTariExtins.csv')
merged_data = alcohol.merge(contintente, left_on='Entity', right_on='Tari', how='inner')
data_for_Ward = alcohol.iloc[:,2:]
data_for_Ward.to_csv('dataOUT/dataWard.csv', index=False)
linkage_matrix = cluster_Ward(data_for_Ward)
data = data_for_Ward.fillna(0)
# componenta_partitiei_optimala(linkage_matrix, alcohol)
partitia_optimala_dendograma(linkage_matrix,data, alcohol_countryindex)
