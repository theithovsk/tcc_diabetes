# imports
import requests
import pandas as pd
import numpy as np
#from datetime import datetime

# documentation
# https://nightscout.github.io/
# https://github.com/nightscout/cgm-remote-monitor/wiki
# http://www.nightscout.info/
# api documentação https://nsday.fly.dev/api-docs
# configurações de visualização
pd.set_option('display.max_columns', None)

# clientes
customers= ['']
#removido por privacidade

# dataframe com os dados brutos de todos os customers referente ao período desejado
train_data = pd.DataFrame()
test_data = pd.DataFrame()


# 
for customer in customers:

    # coleta dos dados de treino
    customer_url = f'https://{customer}.fly.dev/api/v1/entries/sgv.json?find[dateString][$gte]=2023-06-01&find[dateString][$lte]=2023-07-12&find[type]=sgv&count=100000'
    # 
    response = requests.get(customer_url)
    response = response.json()
    # 
    print(f'para o cliente {customer}, coleta de {len(response)} dados de treino;')
    # 
    customer_df = pd.DataFrame(response)
    customer_df['day'] = customer_df.dateString.str.split('T', expand = True)[0]
    customer_df['customer'] = customer
    # 
    train_data = pd.concat([train_data, customer_df])

    # coleta dos dados de teste
    customer_url = f'https://{customer}.fly.dev/api/v1/entries/sgv.json?find[dateString][$gte]=2023-07-13&find[dateString][$lte]=2023-07-14&find[type]=sgv&count=10000'
    # 
    response = requests.get(customer_url)
    response = response.json()
    # 
    print(f'para o cliente {customer}, coleta de {len(response)} dados de teste;')
    # 
    customer_df = pd.DataFrame(response)
    customer_df['day'] = customer_df.dateString.str.split('T', expand = True)[0]
    customer_df['customer'] = customer
    # 
    test_data = pd.concat([test_data, customer_df])

# 
train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')

# 
train_data['hipo severa'] = train_data.sgv.apply(lambda x: True if x <= 54 else False)
train_data['hipo'] = train_data.sgv.apply(lambda x: True if x > 54 and x < 70 else False)
train_data['alvo'] = train_data.sgv.apply(lambda x: True if x >= 70 and x <= 180 else False)
train_data['hiper'] = train_data.sgv.apply(lambda x: True if x > 180 else False)

# 
train_data = train_data[['sgv', 'dateString', 'day', 'customer', 'hipo severa', 'hipo', 'alvo', 'hiper']]

# calcular as frequências por dia nos dados de treino


# filtrar valores absurdos
train_data.query('sgv > 35', inplace = True)
train_data.query('sgv < 400', inplace = True)

# 
train_data_gr = train_data.groupby('day').agg({'sgv': ['mean', 'median', np.std, (lambda x: np.quantile(x, .25)), (lambda x: np.quantile(x, .75))], 'hipo severa': 'sum', 'hipo': 'sum', 'alvo': ['sum', 'count'], 'hiper': 'sum'}).reset_index()

# ajuste dos nomes das colunas
train_data_gr.columns = [f'{col[0]}' if col[1] in ['', 'sum'] else f'{col[0]}_{col[1]}' for col in train_data_gr.columns]

train_data_gr.rename(columns = {'sgv_<lambda_0>': 'sgv_q25', 'sgv_<lambda_1>': 'sgv_q75'}, inplace = True)

# normalização para as agregações de sgv
for col_name in ['sgv_mean', 'sgv_median', 'sgv_std', 'sgv_q25', 'sgv_q75']:
    col_name_min = train_data_gr[col_name].min()
    col_name_max = train_data_gr[col_name].max()
    train_data_gr[col_name] = train_data_gr[col_name].apply(lambda x: (x - col_name_min) / (col_name_max - col_name_min))


# normalização para o tempo no alvo
for col_name in ['hipo severa', 'hipo', 'alvo', 'hiper']:
    train_data_gr[col_name] /= train_data_gr.alvo_count

# 
train_data_gr.drop(columns = 'alvo_count', inplace = True)


#############################################
#           elbow technique
#############################################
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# lista com todas as inércias
inertias = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

# 
for n_clusters in range(2, 21):
    # definição do modelo considerando n_clusters
    n_model = KMeans(n_clusters = n_clusters)
    # treino propriamente dito
    n_model.fit(train_data_gr[[col for col in train_data_gr.columns if col != 'day']])
    # 
    if n_clusters == 6:
        model = n_model
    # inertia é o wscc (within cluster sum of squares)
    inertias.append(n_model.inertia_)
    # cálculos das métricas de eficiência do modelo
    # it varies between -1 and 1; the higher the better;
    score_of_silhouette = silhouette_score(train_data_gr[[col for col in train_data_gr.columns if col != 'day']], n_model.labels_)
    # 
    silhouette_scores.append(score_of_silhouette)
    # the davies-bouldin index measures the average similarity between each cluster and its most similar cluster; lower values indicate better-defined clusters; the dbi ranges from 0 to infinity;
    score_of_davies_bouldin = davies_bouldin_score(train_data_gr[[col for col in train_data_gr.columns if col != 'day']], n_model.labels_)
    # 
    davies_bouldin_scores.append(score_of_davies_bouldin)
    # the calisnki-harabasz index measures the ratio of betewwn-cluster dispersion to within-cluster dispersion; higher values indicate better-defined and more compact clusters; 
    score_of_calinski_harabasz = calinski_harabasz_score(train_data_gr[[col for col in train_data_gr.columns if col != 'day']], n_model.labels_)
    # 
    calinski_harabasz_scores.append(score_of_calinski_harabasz)


# 
from matplotlib import pyplot as plt

# 
elbow_scores_lists = [[inertias, 'Elbow analysis for choosing number of clusters;', 'Inertia', 'Number of clusters'], [silhouette_scores, 'Silhouette score analysis for choosing number of clusters;', 'Silhouette score', 'Number of clusters'], [davies_bouldin_scores, 'Davies bouldin score analysis for choosing number of clusters;', 'Davies bouldin score', 'Number of clusters'], [calinski_harabasz_scores, 'Calinski harabasz score analysis for choosing number of clusters;', 'Calinski harabasz score', 'Number of clusters']]

for analysis in elbow_scores_lists:
    # 
    fig = plt.figure(figsize = (12, 7))
    # 
    plt.scatter([analysis[0].index(el) + 2 for el in analysis[0]], analysis[0])
    # 
    plt.title(analysis[1])
    plt.ylabel(analysis[2])
    plt.xlabel(analysis[3])
    # 
    plt.xticks([analysis[0].index(el) + 2 for el in analysis[0]])

# ajustar a escala em x para ser inteira

#############################################
#       análise do modelo com 6 clusters
#############################################

# 
train_data_gr['cluster'] = model.predict(train_data_gr[[col for col in train_data_gr.columns if col != 'day']])
# 
model.cluster_centers_

#  
# %matplotlib widget
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (12, 7))

if False:
    plt.scatter(train_data_gr.query('cluster == 0').day, train_data_gr.query('cluster == 0').sgv_mean)
    plt.scatter(train_data_gr.query('cluster == 1').day, train_data_gr.query('cluster == 1').sgv_mean)
    plt.scatter(train_data_gr.query('cluster == 2').day, train_data_gr.query('cluster == 2').sgv_mean)

plt.scatter(train_data_gr.query('cluster == 0').sgv_std, train_data_gr.query('cluster == 0').sgv_mean, label = '0')
plt.scatter(train_data_gr.query('cluster == 1').sgv_std, train_data_gr.query('cluster == 1').sgv_mean, label = '1')
plt.scatter(train_data_gr.query('cluster == 2').sgv_std, train_data_gr.query('cluster == 2').sgv_mean, label = '2')
plt.scatter(train_data_gr.query('cluster == 3').sgv_std, train_data_gr.query('cluster == 3').sgv_mean, label = '3')
plt.scatter(train_data_gr.query('cluster == 4').sgv_std, train_data_gr.query('cluster == 4').sgv_mean, label = '4')
plt.scatter(train_data_gr.query('cluster == 5').sgv_std, train_data_gr.query('cluster == 5').sgv_mean, label = '5')
plt.xlabel('sgv_std')
plt.ylabel('sgv_mean')
plt.legend()

if False:
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = Axes3D(fig)
    x = train_data_gr.query('cluster == 0').sgv_mean.values.tolist()
    y = train_data_gr.query('cluster == 0').sgv_median.values.tolist()
    z = train_data_gr.query('cluster == 0').sgv_std.values.tolist()
    sgv_plot = ax.scatter(x, y, z, color = 'orange')
    plt.show()


############################################################
#          aplicando o modelo no conjunto de testes
############################################################

test_data['hipo severa'] = test_data.sgv.apply(lambda x: True if x <= 54 else False)
test_data['hipo'] = test_data.sgv.apply(lambda x: True if x > 54 and x < 70 else False)
test_data['alvo'] = test_data.sgv.apply(lambda x: True if x >= 70 and x <= 180 else False)
test_data['hiper'] = test_data.sgv.apply(lambda x: True if x > 180 else False)

# 
test_data = test_data[['sgv', 'dateString', 'day', 'customer', 'hipo severa', 'hipo', 'alvo', 'hiper']]

# calcular as frequências por dia nos dados de treino


# filtrar valores absurdos
test_data.query('sgv > 35', inplace = True)
test_data.query('sgv < 400', inplace = True)

# 
test_data_gr = test_data.groupby('customer').agg({'sgv': ['mean', 'median', np.std, (lambda x: np.quantile(x, .25)), (lambda x: np.quantile(x, .75))], 'hipo severa': 'sum', 'hipo': 'sum', 'alvo': ['sum', 'count'], 'hiper': 'sum'}).reset_index()

# ajuste dos nomes das colunas
test_data_gr.columns = [f'{col[0]}' if col[1] in ['', 'sum'] else f'{col[0]}_{col[1]}' for col in test_data_gr.columns]

test_data_gr.rename(columns = {'sgv_<lambda_0>': 'sgv_q25', 'sgv_<lambda_1>': 'sgv_q75'}, inplace = True)

# normalização para as agregações de sgv
for col_name in ['sgv_mean', 'sgv_median', 'sgv_std', 'sgv_q25', 'sgv_q75']:
    col_name_min = test_data_gr[col_name].min()
    col_name_max = test_data_gr[col_name].max()
    test_data_gr[col_name] = test_data_gr[col_name].apply(lambda x: (x - col_name_min) / (col_name_max - col_name_min))


# normalização para o tempo no alvo
for col_name in ['hipo severa', 'hipo', 'alvo', 'hiper']:
    test_data_gr[col_name] /= test_data_gr.alvo_count

# 
test_data_gr.drop(columns = 'alvo_count', inplace = True)

# aplicação propriamente dita do modelo nos dados de teste
test_data_gr['cluster'] = model.predict(test_data_gr[[col for col in test_data_gr.columns if col not in  ['customer']]])

# criação do plot dos resultados de teste
fig = plt.figure(figsize = (12, 7))

if False:
    plt.scatter(test_data_gr.query('cluster == 0').day, test_data_gr.query('cluster == 0').sgv_mean)
    plt.scatter(test_data_gr.query('cluster == 1').day, test_data_gr.query('cluster == 1').sgv_mean)
    plt.scatter(test_data_gr.query('cluster == 2').day, test_data_gr.query('cluster == 2').sgv_mean)

plt.scatter(test_data_gr.query('cluster == 0').hiper, test_data_gr.query('cluster == 0').sgv_std, label = '0')
plt.scatter(test_data_gr.query('cluster == 1').hiper, test_data_gr.query('cluster == 1').sgv_std, label = '1')
plt.scatter(test_data_gr.query('cluster == 2').hiper, test_data_gr.query('cluster == 2').sgv_std, label = '2')
plt.scatter(test_data_gr.query('cluster == 3').hiper, test_data_gr.query('cluster == 3').sgv_std, label = '3')
plt.scatter(test_data_gr.query('cluster == 4').hiper, test_data_gr.query('cluster == 4').sgv_std, label = '4')
plt.scatter(test_data_gr.query('cluster == 5').hiper, test_data_gr.query('cluster == 5').sgv_std, label = '5')
plt.xlabel('hiper')
plt.ylabel('sgv_std')
plt.legend()
#plt.title(f'Análise de resultados do dia {test_data.day.unique().tolist()[0]}')
