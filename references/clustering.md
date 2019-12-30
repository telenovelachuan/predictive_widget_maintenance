# Anomaly Detection

Try clustering the unit data samples on different granularities and visualizing the output.

# Method 1: cluster samples based on sample point level.

1. Merging all unit data files. Run PCA to check explained variance ratio

![pca_exp_ratio](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_exp_ratio.png)
The original dataset could be condensed into 2 - 3 dimensions with over 95% variance preserved.

2. Squash the dataset into 2 dimensions for visualization, each color representing a unit.
![pca_2_units](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_2_units.png)
The unit file patterns are quite separate, while each one being in a similar Baguette shape.

3. Try t-SNE (t-distributed stochastic neighbor embedding) for dimensionality reduction and visualization.
![tsne_2_units](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/tsne_2_units.png)
Compared to PCA, t-SNE plots the data points more chaotically without a clear boundary for each unit file.

4. Try DBSCAN to cluster all data samples in every unit file.
![pca_dbscan](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_2_DBSCAN.png)
As shown above, DBSCAN did not form evenly distributed clusters in our case. Most points fall into the first cluster.

5. Try K-Means clustering
![pca_kmeans_10](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_Kmeans_10.png)
KMeans with 10 clusters generates more balanced clusters, with each one scattering hierarchically inside each data file.

![pca_kmeans_15](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_Kmeans_15.png)
For KMeans with 15 clusters, the cluster sizes polarized.

# Method 2: cluster samples based on unit level.

1. Aggregating stats descriptions of all unit files, and merging them into a training set.
![method2_ds](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/method2_dataset.png)

2. Run Kmeans on the unit level training data.
![kmeans_train](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/kmeans_train.png)
It can be obviously observed that there are 2 cluster for the stats description.

cluster 1: 0000, 0001, 0003, 0004, 0005, 0006, 0007, 0010, 0011, 0012, 0016, 0017, 0018, 0019

cluster 2: 0002, 0008, 0009, 0014, 0015

cluster 3: 0013

3. Run Kmeans on all the test data and train data.
![kmeans_all](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/kmeans_all.png)
Just like training dataset, it's lucky that the test set also splits into these 2 clusters.

cluster 1: 0020, 0021, 0022, 0024, 0025, 0026, 0027, 0028, 0029, 0030, 0031, 0032, 0033, 0034, 0035, 0036, 0037, 0038, 0039, 0040, 0041, 0042, 0043, 0044, 0045, 0046, 0047, 0048, 0049

cluster 2: 0023














