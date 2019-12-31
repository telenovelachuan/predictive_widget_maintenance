A data science project on the widget maintenance dataset of ExampleCo, Inc to explore feature trends of its data generating system and predict the units that are mostly likely to fail.

## Feature exploring

- rpm of unit 0010:
![rpm_10](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/rpm_10.png)

- motor_voltage of various units
![motor_voltages](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/motor_voltages.png)

- inlet temp of various units
![inlet_temps](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/inlet_temps.png)

- rpm distribution of all units
![rpm_distrs](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/rpm_distrs.png)
 
- motor_current distribution of all units
![motor_currents](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/motor_current_distr.png)
 
-  Pearson correlation between features of various units
![pearson_corrs](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/Pearson_correlations.png)

## Excluding Anomalies

Conduct anomaly detection and removing anomalies on the train data and test data.

###### method 1: use boxplot to peek data range and detect anomalies. Below is an example on unit 0001.

![boxplot1](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/boxplots_1.png)

And after anomalies are excluded:
![rpm_boxplot](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/rpm_1_wo_anml.png)
 
###### method 2: try isolation forest for detecting anomalies.

After excluding anomalies using isolation forest, the rpm of unit 0001 looks like:
![rpm_if](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/rpm_1_wo_anml_IF.png)

Apply the first method and exclude the anomalies in all units. Most of the outliers are removed and the patterns look more concentrated. Below is an example on unit 0022.

![comp_22](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/comparison_22.png)

## Clustering

Try clustering the unit data samples on different granularities and visualizing the output.

- use PCA to squash the dataset into 2 dimensions for visualization, each color representing a unit. PCA outperforms t-SNE here on dimensionality reduction, with explained variance ratio of over 95%.
![pca_2_units](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_2_units.png)

###### method 1: cluster all data samples in every unit file. KMeans and DBSCAN were used, KMeans generates more evenly distributed clusters across all unit files.
![pca_kmeans_10](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_Kmeans_10.png)

###### method 2: cluster samples based on unit level. The stats descriptions of all unit files are aggregated, and merged into a training set, on which KMeans was run.

Training set clusters generated by KMeans are as below.
![kmeans_train](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/kmeans_train.png)

It can be obviously observed that there are 2 cluster for the stats description.

	- cluster 1: 0000, 0001, 0003, 0004, 0005, 0006, 0007, 0010, 0011, 0012, 0016, 0017, 0018, 0019

	- cluster 2: 0002, 0008, 0009, 0014, 0015

And finally test set joins in.
![kmeans_all](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/kmeans_all.png)

Just like training dataset, it's lucky that the test set also splits into these 2 clusters.

	- cluster 1: 0020, 0021, 0022, 0024, 0025, 0026, 0027, 0028, 0029, 0030, 0031, 0032, 0033, 0034, 0035, 0036, 0037, 0038, 0039, 0040, 0041, 0042, 0043, 0044, 0045, 0046, 0047, 0048, 0049

	- cluster 2: 0023
	





