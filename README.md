A data science project on the widget maintenance dataset of ExampleCo, Inc to explore feature trends of its data generating system and predict the units that are mostly likely to fail.

## Feature exploring

Take a look at feature values and their distributions

- rpm of unit 0010:
![rpm_10](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/rpm_10.png)

- motor_voltage of various units
![motor_voltages](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/motor_voltages.png)

- motor_temp of various units
![motor_temps](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/motor_temps.png)

- rpm distribution of all units
![rpm_distrs](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/rpm_distrs.png)
 
- motor_current distribution of all units
![motor_current_distrs](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/motor_current_distr.png)
 
- inlet_temp distribution of all units
![inlet_temp_distrs](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/inlet_temp_distrs.png)
 
An obvious pattern observed here is that in every feature, there're 2 groups of distribution pattern existing in all units. This takes us to an initiative of clustering attempts.

-  Pearson correlation between features of various units
![pearson_corrs](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/feature_visualization/Pearson_correlations.png)

It can be observed that the feature "rpm" is quite separate from all other features, while "motor_voltage", "motor_current", "motor_temp" and "inlet_temp" are somewhat correlated in most units.

[Click me for more details of feature visualization work](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/notebooks/feature%20visualization.ipynb)

## Excluding Anomalies

Conduct anomaly detection and removing anomalies on the train data and test data.

###### method 1: use boxplot to peek data range and detect anomalies. Below is an example on unit 0001.

![boxplot1](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/boxplots_1.png)

And after anomalies are excluded:
![rpm_boxplot](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/rpm_1_wo_anml.png)
 
###### method 2: try isolation forest for detecting anomalies.

After excluding anomalies using isolation forest, the rpm of unit 0001 looks like:
![rpm_if](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/rpm_1_wo_anml_IF.png)

Method 1 seems to remove more reasonable outliers and normalize the distribution better. So apply the first method here to exclude anomalies in all units. Most of the outliers are removed and the patterns look more concentrated. Below is an example of before and after excluding anomalies on unit 0001.

![comp_1](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/comparison_1.png)

[Click me for more details of anomaly detection work](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/references/anomaly_detection.md)

## Time Series Analysis

Try to predict the feature trends of a unit going with time.

- Look at the rolling mean & std of feature motor_voltage in all units
![motor_voltage_rollings](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/time_series/motor_voltage_rollings.png)

###### Try to predict the trend of motor_voltage for unit 0019.

- use AD-Fuller test for confirming stationarity
	- ADF Statistic: -8.357128
	- p-value: 0.000000
	
The p-value is smaller than threshold 0.05, which indicates data stationarity.

- peek at the lately trend of its motor_voltage.
![motor_voltage_19_lately](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/time_series/motor_voltage_19_lately.png)

- plot the autocorrelation & partial autocorrelation of motor_voltage for unit 0019
![motor_voltage_19_pacf](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/time_series/motor_voltage_19_pacf.png)

- train seasonal ARIMA model and predict the future values with confidence interval.

Train SARIMA model for unit 0019 motor_voltage. Differencing is set to 0 due to data stationarity, while p and q are both set since there're cut-offs in autocorrelation and partial autocorrelation. Grid search is used to find the best param combinations. And finally use the trained SARIMA to predict the future 15% data.
![motor_voltage_19_SARIMAX](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/time_series/motor_voltage_19_SARIMAX.png)

###### Another attempt: to predict the trend of feature "motor_temp" on unit 0018.

- the original data trend looks like:
![motor_temp_18](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/time_series/motor_temp_18.png)

- train SARIMAX model on the lately trend based on acf, pacf and grid search, and predict the future values with confidence interval.
![motor_temp_18_SARIMAX](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/time_series/motor_temp_18_SARIMAX.png)

[Click me for more details of time series analysis work](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/notebooks/time%20series%20analysis.ipynb)


## Clustering

Try clustering the unit data samples on different granularities and visualizing the output.

- use PCA to squash the dataset into 2 dimensions for visualization, each color representing a unit. PCA outperforms t-SNE here on dimensionality reduction, with explained variance ratio of over 95%.
![pca_2_units](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_2_units.png)

###### method 1: cluster all data samples in every unit file. KMeans and DBSCAN were used, KMeans generates more evenly distributed clusters across all unit files.
![pca_kmeans_10](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/PCA_Kmeans_10.png)

###### method 2: cluster samples based on unit level. The stats descriptions of all unit files are aggregated, and merged into a training set, on which KMeans was run.

Training set clusters generated by KMeans are as below.
![kmeans_train](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/kmeans_train.png)

It can be observed that there are 2 cluster for the stats description.

	cluster 1: 0000, 0001, 0003, 0004, 0005, 0006, 0007, 0010, 0011, 0012, 0016, 0017, 0018, 0019

	cluster 2: 0002, 0008, 0009, 0014, 0015

And finally test set joins in.
![kmeans_all](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/clustering/kmeans_all.png)

Just like training dataset, it's lucky that the test set also splits into these 2 clusters.

	cluster 1: 0020, 0021, 0022, 0024, 0025, 0026, 0027, 0028, 0029, 0030, 0031, 0032, 0033, 0034, 0035, 0036, 0037, 0038, 0039, 0040, 0041, 0042, 0043, 0044, 0045, 0046, 0047, 0048, 0049

	cluster 2: 0023

[Click me for more details of clustering work](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/references/clustering.md)
	
## Failure Predicting

Try to predict the units that are most likely to fail in near future.

###### Idea 1: compute the days until failure for each data sample, and train regression MLP to predict the days until failure for test set. So every prediction on test sample tells the time it predicts to fail. Aggregating all predictions would just "vote" for the probability of the unit failure within a given time period.

1. Preparing 1 label and 2 new features before training model:

	- time remaining until failure (in days)

	- accumulated warnings generated till now

	- accumulated errors generated till now

After combining features with alarming data, let's try to plot some feature trends (Yellow lines are warnings and red lines are errors):
![motor_voltage_18](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/motor_voltage_18.png)

2. Normalizing all feature values and y labels. Below is the distribution of y label for training set.
![y_train_distr](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/y_train_distr.png)

3. Constructing MLP model using Tensorflow Keras API.

After tons of paramter tuning, it turned out that Dense layers with Lecun normal init and Adadelta optimizer, followed by leaky-ReLU activation outperformed all the other decent options. Model training early stopped at .58 r square, where overfitting began to populate.

![history_all](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/training_history_all.png)

4. Use the trained model to predict the days until failure for new units. Compute the possibility of failure within next 30 days and average failure time predicted by the model on every row.

From the model predictions, the units with largest failure probability in the next 30 days are:

	- unit 39: 0.178

	- unit 47: 0.188

	- unit 21: 0.276

	- unit 28: 0.294

	- unit 27: 0.337

	- unit 40: 0.349


###### Idea 2: train regression MLP for the two groups(clustered by Kmeans) respectively, and predict according to clusters.

1. Train-test split the merged data of units in cluster 1, normalizing all inputs and label values.

2. Construct Keras neural network for the dataset on cluster 1, using 5 dense layers, leaky ReLU activation, Adadelta optimization and Lecun init(tuned out to be the optimal). R square is used for evaluation metric.
![history_c1](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/training_history_c1.png)

The training process achieved .66 r square on testing set before early stopping detected overfitting.

3. Aggregate data and train-test split the merged data of units in cluster 2, normalizing all inputs and label values.

4. Construct Keras neural network for the dataset on cluster 2. Less layers and nerons are needed since training data becomes less. Leaky-ReLU and Adadelta optimizer still outperformed all the other options.
![history_c2](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/training_history_c2.png)

5. For the test units from 20 to 49, find their corresponding model based on cluster and predict the failure probabilities. The units with biggest failure probability in the next 30 days are:

	- unit 42: 0.122

	- unit 21: 0.129

	- unit 28: 0.131

	- unit 32: 0.176

	- unit 40: 0.191

	- unit 27: 0.201

	- unit 23: 0.523
	
Besides, I also tried random forest regressor on the cluster 2 data. 400 estimators reached an R square of around .55 on 3-fold cross validation, which was a little bit better than MLP but not a dramatic improvement, so finally I used MLP for final predicting.








