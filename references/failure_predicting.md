# Failure Predicting
Try to predict which units are most possible to fail soon.

# Method 1: use neural network and train MLP on all unit training data files

1. Preparing 3 new feature for each data row before training model:

	- time remaining until failure (in days)

	- accumulated warnings generated till now

	- accumulated errors generated till now
	
After combining features with alarming data, let's try to plot some feature trends:
![motor_voltage_18](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/motor_voltage_18.png)

2. Normalizing all feature values and y labels. Below is the distribution of y label for training set.
![y_train_distr](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/y_train_distr.png)

3. Construct Keras MLP model for data merging all units.

After tons of paramter tuning, it turned out that Dense layers of Lecun normal init and Adadelta optimizer, followed by leaky-ReLU activation outperformed all the other decent options. Model training early stopped at .58 r square, where overfitting began to populate.
![history_all](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/training_history_all.png)

4. Use the trained model to predict the days until failure for new units. Compute the possibility of failure within next 30 days and average failure time predicted by the model on every row.

From the model predictions, the units with largest failure probability in the next 30 days are:

	- unit 39: 0.178

	- unit 47: 0.188

	- unit 21: 0.276

	- unit 28: 0.294

	- unit 27: 0.337

	- unit 40: 0.349


# Method 2: train MLP for the two groups(clustered by Kmeans) respectively

1. Train-test split the merged data of units in cluster 1, normalizing all inputs and label values.

2. Construct Keras neural network for the dataset on cluster 1, using 5 dense layers, leaky ReLU activation, Adadelta optimization and Lecun init(tuned out to be the optimal). R square is used for evaluation metric.
![history_c1](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/training_history_c1.png)

The training process achieved .66 r square on testing set before early stopping detected overfitting.

3. Aggregate data and train-test split the merged data of units in cluster 2, normalizing all inputs and label values.

4. Construct Keras neural network for the dataset on cluster 2. Less layers and nerons are needed since training data becomes less. Leaky-ReLU and Adadelta optimizer still outperformed all the other options.
![history_c2](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/failure_predicting/training_history_c2.png)

5. From the 2 models for cluster 1 and 2, the units with largest failure probability in the next 30 days are:

	- unit 42: 0.122

	- unit 21: 0.129

	- unit 28: 0.131

	- unit 32: 0.176

	- unit 40: 0.191

	- unit 27: 0.201

	- unit 23: 0.523
	
Besides, I also tried random forest regressor on the cluster 2 data. 400 estimators reached an R square of around .55 on 3-fold cross validation, which was a little bit better than MLP but not a dramatic improvement, so finally I used MLP for final predicting.


