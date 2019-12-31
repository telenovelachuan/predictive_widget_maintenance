import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from joblib import dump, load
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import nbformat
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

TEMP_LOW_LIMIT = -459.67
TEMP_HIGH_LIMIT = 1000
RPM_LOW_LIMIT = 0
RPM_HIGH_LIMIT = 6e8  # the largest achieved rpm of all human artifacts


def exclude_physical_anml(df):
    result = df[(df['motor_temp'] > TEMP_LOW_LIMIT) & (df['inlet_temp'] > TEMP_LOW_LIMIT) & (df['rpm'] > 0) & (df['rpm'] < RPM_HIGH_LIMIT)]
    return result


def peek_attributes_trend(attribute, ylim_low=-10, ylim_high=1500):
    units = ["000{}".format(num) if num < 10 else "00{}".format(num) for num in random.sample(range(0, 20), 9)]
    file_names = ["../data/raw/train/unit{}_rms.csv".format(unit) for unit in units]
    sns.set()
    f, axs = plt.subplots(3, 3, figsize=(20, 12))
    for idx, file_name in enumerate(file_names):
        file_df = pd.read_csv(file_name)
        file_df = exclude_physical_anml(file_df)
        ax1 = plt.subplot(3, 3, idx + 1)
        file_df[attribute].plot(ylim=(ylim_low, ylim_high))
    plt.show()


def load_file(unit, file_type='rms', version='raw', folder='train'):
    unit_name = "000{}".format(unit) if unit < 10 else "00{}".format(unit)
    if version == 'raw':
        file_name = '../data/raw/{}/unit{}_{}.csv'.format(folder, unit_name, file_type)
    else:
        file_name = '../data/processed/{}/unit{}_{}_anomaly_excluded.csv'.format(folder, unit_name, file_type)
    df = pd.read_csv(file_name, index_col=0)
    if file_type == 'rms':
        df = exclude_physical_anml(df)
    return df


def load_all_files():
    units = range(0, 20)
    df_all = pd.DataFrame()
    for idx, unit in enumerate(units):
        file_name = "../data/processed/train/unit{}_rms_anomaly_excluded.csv".format("000{}".format(unit) if unit < 10 else "00{}".format(unit))
        file_df = pd.read_csv(file_name, index_col=0)
        file_df['unit'] = unit
        df_all = df_all.append(file_df)
    return df_all


def transform_column(df_input):
    num_features = df_input.columns[1:]
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_features)
    ])
    np_pipelined = full_pipeline.fit_transform(df_input)
    df_pipelined = pd.DataFrame(np_pipelined, index=df_input.index)
    return df_pipelined


def count_clustered_labels(labels_, algorithm_name):
    print("clustering result by {}\n{}".format(algorithm_name, Counter(labels_)))


df_all = load_all_files()


def run_PCA(df_input, n_components=6):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(transform_column(df_input))
    print(pca.explained_variance_ratio_)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Explained variance ratio")
    plt.title("Explained Variance")
    plt.xticks(np.arange(0, n_components, 1))
    plt.show()
    return pca_result


pca_6_all_result = run_PCA(df_all, 6)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(transform_column(df_all))
fig, ax = plt.subplots(figsize=(13, 13))
plt.scatter(x=[p[0] for p in pca_result], y=[p[1] for p in pca_result], c=df_all['unit'], cmap='Spectral', marker='o', alpha=0.5, s=10)
plt.show()

tsne_model_all = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_vectors_all = tsne_model_all.fit_transform(transform_column(df_all))

fig, ax = plt.subplots(figsize=(13, 13))
tsne_vectors_all_scaled = tsne_vectors_all * 1e5
plt.scatter(x=tsne_vectors_all_scaled[:, 0], y=tsne_vectors_all_scaled[:, 1], marker='o', c=df_all['unit'], cmap='Spectral', alpha=0.5, s=10)
plt.show()

dbscan = DBSCAN(eps=15, min_samples=5)
print("model initialized")
dbscan.fit(df_all[df_all.columns[1:]])
print("model trained")
dump(dbscan, '../models/clustering/all_units_dbscan.joblib')
print("model saved")


def plot_clustering(algorithm_name, labels, pca_result, size=5, alpha=0.5, cmap='Spectral'):
    print("clustering result by {}\n{}".format(algorithm_name, Counter(labels)))
    fig, ax = plt.subplots(figsize=(13, 13))
    plt.scatter(x=[p[0] for p in pca_result], y=[p[1] for p in pca_result], c=labels, cmap=cmap, marker='o', alpha=alpha, s=size)
    plt.show()


dbscan = load('../models/clustering/all_units_dbscan.joblib')
plot_clustering('DBSCAN', dbscan.labels_, pca_result)


def train_kmeans(n_clusters=10, init='k-means++', n_init=10, tol=1e-4, precompute_distances='auto', algorithm='auto'):
    df_all_features = df_all.drop(columns=['timestamp'], axis=1)
    kmeans_model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, tol=tol,
                   precompute_distances=precompute_distances, algorithm=algorithm)
    kmeans_model.fit_predict(df_all_features)
    return kmeans_model


km_10 = train_kmeans(n_clusters=10)
plot_clustering('KMeans with default parameters', km_10.labels_, pca_result)

km_15 = train_kmeans(n_clusters=15, n_init=20, precompute_distances=True, algorithm='elkan')
plot_clustering('KMeans with 15 clusters', km_15.labels_, pca_result)

columns = load_file(0).columns.tolist()
stat_attrs = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
for element in itertools.product(columns, stat_attrs):
    print(element)


def aggregate_stats_all_units(folder='train'):
    columns = load_file(0).columns.tolist()
    stat_attrs = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    df_all_stats_columns = []
    for element in itertools.product(columns, stat_attrs):
        df_all_stats_columns.append("{}-{}".format(element[0], element[1]))
    df_all_stats = pd.DataFrame(columns=df_all_stats_columns)

    units = range(0, 20) if folder == 'train' else range(20, 50)
    for unit in units:
        df_unit = load_file(unit, version='processed', folder=folder).describe()
        row_value_dict = {}
        for column_name in df_all_stats_columns:
            feature, stat_attr = column_name.split('-')
            row_value_dict[column_name] = df_unit.loc[stat_attr][feature]
            df_all_stats.loc[unit] = pd.Series(row_value_dict)
    return df_all_stats


df_all_stats_train = aggregate_stats_all_units(folder='train')

km_2_train_unit_level = KMeans(n_clusters=3, n_init=20, precompute_distances=True, algorithm='elkan')
km_2_train_unit_level.fit_predict(df_all_stats_train)
plot_clustering('KMeans on unit level data', km_2_train_unit_level.labels_, pca_2_train_result, size=30, alpha=1, cmap='viridis')

df_all_stats_test = aggregate_stats_all_units(folder='test')
df_all_stats = pd.concat([df_all_stats_train, df_all_stats_test])

pca_2_all = PCA(n_components=2)
pca_2_all_result = pca.fit_transform(transform_column(df_all_stats))
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(x=[p[0] for p in pca_2_all_result], y=[p[1] for p in pca_2_all_result], cmap='Spectral', marker='o', alpha=1, s=30)
plt.show()

km_2_all_unit_level = KMeans(n_clusters=3, n_init=20, precompute_distances=True, algorithm='elkan')
km_2_all_unit_level.fit_predict(df_all_stats)
plot_clustering('KMeans on unit level data', km_2_all_unit_level.labels_, pca_2_all_result, size=30, alpha=1, cmap='viridis')

