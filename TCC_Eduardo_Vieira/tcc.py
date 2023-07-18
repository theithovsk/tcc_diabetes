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
customers = ['nsday', 'nssamara', 'nsjordana', 'nsarnaldo', 'nsaisha', 'nscamila', 'nsdaniele', 'nsariane', 'nsanaxavier', 'samuelgarcia2011', 'giovannaruggiero', 'miaomiaogisele', 'luishenriquedm1', 'nsaldicleide', 'miaomiaofabiana', 'jvnightscout', 'nsjoao', 'nsalyson', 'nspanassi', 'nsdaiana', 'alicelyra', 'glicemiaslavinia', 'nsraquel', 'gabrielhrocha', 'nskarla', 'njorge', 'ayslan', 'leticia', 'nightscoutnery', 'enzodetoledo', 'camila', 'ariele', 'nsfernando', 'nselise', 'miaomiaojoana', 'nsraissa', 'ccwerner', 'luizabfroes', 'anabeatriz', 'nsandreia', 'nspaula', 'nstaysla', 'vitorpassos', 'nsrose', 'nslavinia', 'nsguilherme', 'nsfabiola', 'lauraspinh0', 'cgmleonardo', 'nsluciano', 'nsduarte', 'nightscoutadriana', 'isabelamota', 'nightscoutderi', 'nstatiane', 'vividm', 'nshelenice', 'miaomiaomaria', 'nsjeferson', 'nscecilia', 'nightscoutedilene', 'glicemianathan', 'nsgisellem', 'nsamanda', 'alinebrag', 'glicemiafabiana', 'nsrita', 'mancinimiguel', 'diabetestipo1prv', 'nightscoutluciana', 'angelina2017dm1', 'catarinagerotto', 'luccafabiano', 'benicio', 'gabrieldamascena', 'nightscoutluiz', 'nskelly', 'nsgabriela', 'nightscoutangela', 'nightscoutaline', 'nsthaizy', 'nightscoutsueli', 'nightscoutkaty', 'nightscoutwelen', 'rayanne', 'miaomiaohellen', 'kellenfernanda', 'nightscoutpriscila', 'nightscoutrodrigo', 'nightscoutancelmo', 'nightscoutjulia', 'nsmiriam', 'nsadriana', 'barbaragomespca', 'emanupfsilva', 'nightscoutmirela', 'grazielee', 'nightscoutrenata', 'luisasantos', 'samuelmuzi', 'nightscoutlili', 'joaopedrojp', 'gabrieldm1', 'hadassa', 'nsfatima', 'nightscouttipaldi', 'claramendesdm1', 'nsjoana', 'nstalita', 'nspedro', 'nslarissa', 'emanuelle', 'waniadias', 'miaomiaovalentina', 'guilhermelucas2015', 'miaomiaojose', 'eduardalhul12', 'icaroguilherme', 'alicedocinho', 'anakarine', 'nsluciana', 'nsalinecardoso', 'nspaulabosio', 'nsdamicheline', 'nsmairteixeira', 'nscamilap', 'nscarlos', 'nsrenato', 'nslorena', 'nsanaluiza', 'nselaine', 'nsgraziela', 'nsacacio', 'nsgabriel', 'nsjosecarlos', 'nskristine', 'thaleslourenci', 'nsveri', 'alice', 'nseulimar', 'nsedivania', 'nsdaguia', 'nseuclides', 'nsanacristina', 'nsrenatasousa', 'nightscoutjosy', 'nsinaura', 'nsarmenia', 'nsluana', 'nsbernardo', 'joaopinheiro', 'nsfabianaalves', 'nsarthur', 'nsmarciele', 'nsiracema', 'nsnoadia', 'nsdebs', 'nsestefania', 'nskenia', 'nsdanila', 'nsalexandre', 'nspaulapenna', 'nsnice', 'nsfabricia', 'nsandressa', 'nightscoutnatalia', 'nssheila', 'nspalmeida', 'nsneusa', 'nscarol', 'nslucianamaia', 'nsevandro', 'nightscoutclaudio', 'nsambiel', 'nsmaneca', 'nssalome', 'nsvanessamiranda', 'aninhafree', 'nslucasnogueira', 'nsalinepaiva', 'clarinha', 'rodabete', 'nightscoutfatima', 'nsalexandref', 'nanda10', 'arthur03', 'nsmelissa', 'nspittol', 'nsduanne', 'nsmarcia', 'nsangelo', 'nsrafaelsantos', 'nsleila', 'nslaura', 'nsdanimonteiro', 'nslucianaaguiar', 'nscarina', 'nselena', 'nsanapaula', 'nsleticiab', 'nssaraalves', 'nskarina', 'nsmariana', 'nightscouterika', 'nspoliana', 'nsmariapaula', 'nseuza', 'nsandrezza', 'nsmary', 'nsdavi', 'nsisadorasantos', 'nsizabela', 'nsgabrielad', 'nsrogerio', 'nsthielly', 'nightscoutdaniele', 'nspolianareinert', 'nsreginaldo', 'nspatricianobrega', 'nsluisamacedo', 'nsgraciela', 'nsmalheiros', 'nsdanielesilva', 'nsgeorgina', 'nsmarilda', 'talitadm1', 'nsclaudialopes', 'nsrhaissa', 'nsrosmar', 'nsdaianecalera', 'nsrenata', 'nsleolaine']
#customers= ['nsday']

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

# treinar modelo de classificação
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
# treino propriamente dito
model.fit(train_data_gr[[col for col in train_data_gr.columns if col != 'day']])
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
plt.xlabel('hiper')
plt.ylabel('sgv_std')
plt.legend()
#plt.title(f'Análise de resultados do dia {test_data.day.unique().tolist()[0]}')

