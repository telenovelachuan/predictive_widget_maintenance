# Anomaly Detection

Conducting anomaly detection and removing anomalies on the train data and test data.

1. Remove some unreal data points (physically impossible temperature, rpm, etc.).

2. Take a look at the value distributions of some features in different units.

rpm:
![rpms](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/raw_rpms.png)

3. Use boxplot to compute anomaly values for a unit.

Boxplots for features of unit 0001:
![boxplot1](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/boxplots_1.png)

After excluding anomalies using boxplot, the motor current of unit 0001 looks like:
![rpm_boxplot](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/rpm_1_wo_anml.png)


4. Try isolation forest for detecting anomalies.

After excluding anomalies using isolation forest, the rpm of unit 0001 looks like:
![rpm_if](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/rpm_1_wo_anml_IF.png)


5. Apply the first method and exclude the anomalies in all units. Compare the difference between before exlcuding anomalies and after exlcuding anomalies. Mostly of the outliers are removed and the patterns look more concentrated.

Before and after excluding anomalies on unit 0001:
![comp_1](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/comparison_1.png)

Before and after excluding anomalies on unit 0022:
![comp_22](https://github.com/telenovelachuan/predictive_widget_maintenance/blob/master/reports/figures/anomaly_detection/comparison_22.png)

