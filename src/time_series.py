import pandas as pd
import numpy as np
import random
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import timedelta
import itertools
import warnings
import matplotlib.dates as mdate

warnings.filterwarnings("ignore")
Y_LIMS = {
    'rpm': (-20, 1500),
    'motor_voltage': (0, 400),
    'motor_current': (-20, 100),
    'motor_temp': (-20, 250),
    'inlet_temp': (-20, 250),
}

def load_unit_data(unit):
    unit_name = "000{}".format(unit) if unit < 10 else "00{}".format(unit)
    file_df = pd.read_csv("../data/processed/train/unit{}_rms_anomaly_excluded.csv".format(unit_name))
    file_df['timestamp'] = pd.to_datetime(file_df['timestamp'])
    file_df.set_index('timestamp', inplace=True)
    return file_df

def get_last_proportion(unit, feature, proportion=0.1):
    df_unit = load_unit_data(unit)
    index_prop = int(len(df_unit.index) * (1- proportion))
    return df_unit[index_prop: ]

def log_data(unit, feature_name, proportion):
    df_data = get_last_proportion(unit, feature_name, proportion)
    return np.log(df_data[feature_name])

def plot_feature_in_single_file(feature_name, unit, ylim_low=None, ylim_high=None, bins=20, show_dist=True):
    file_df = load_unit_data(unit)
    sns.set()
    fig, axs = plt.subplots(2 if show_dist else 1, 1, figsize=(16, 10))
    subplot_idx = 211 if show_dist else 111
    ax1 = fig.add_subplot(subplot_idx)
    if ylim_low is not None or ylim_low is not None:
        file_df[feature_name].plot(ylim=(ylim_low, ylim_high))
    else:
        file_df[feature_name].plot(ylim=Y_LIMS[feature_name])

    #xticks_count = 30
    #xtick_dates_indexes = [int(n) for n in np.arange(0, len(file_df.index), step=len(file_df.index) / xticks_count)]
    #xticks_dates = [datetime.datetime.strptime(file_df.index[i][:19], '%Y-%m-%d %H:%M:%S') for i in xtick_dates_indexes]

    if show_dist == True:
        ax2 = fig.add_subplot(212)
        sns.distplot(file_df[feature_name], bins=bins)
    plt.show()


plot_feature_in_single_file("rpm", 12, ylim_low=800, ylim_high=1300, bins=100, show_dist=True)


def plot_rolling_mean_std(unit, feature, window=12, ylim_low=-10, ylim_high=1500, figsize=None):
    df_unit = load_unit_data(unit)[[feature]]
    rolling_mean = df_unit.rolling(window=window).mean()
    rolling_std = df_unit.rolling(window=window).std()

    plt.figure(figsize=figsize, dpi=100)
    plt.plot(df_unit, color='blue', label='Original')
    plt.plot(rolling_std, color='grey', label='Rolling Std')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.legend(loc='best')
    plt_title = 'Rolling Mean & Rolling std for unit {}'.format(unit)
    plt.title(plt_title)
    plt.show()


plot_rolling_mean_std(0, 'rpm', figsize=(12, 5))


def plot_rolling_meas_stds(feature, window=12, ylim_low=-10, ylim_high=1500):
    units = random.sample(range(0, 20), 9)
    # units = ["000{}".format(num) if num < 10 else "00{}".format(num) for num in random.sample(range(0, 20), 9)]
    file_names = ["../data/processed/train/unit{}_rms_anomaly_excluded.csv".format(unit) for unit in units]
    sns.set()
    f, axs = plt.subplots(3, 3, figsize=(20, 12))
    for idx, unit in enumerate(units):
        ax = plt.subplot(3, 3, idx + 1)
        df_unit = load_unit_data(unit)[[feature]]
        rolling_mean = df_unit.rolling(window=window).mean()
        rolling_std = df_unit.rolling(window=window).std()

        ax.plot(df_unit.index, df_unit[feature], color='blue', label='Original')
        ax.plot(rolling_std, color='grey', label='Rolling Std')
        ax.plot(rolling_mean, color='red', label='Rolling Mean')
        ax.legend(loc='best')
        ax.set_title('Rolling Mean & Rolling std for unit {}'.format(unit))
    plt.show()


plot_rolling_meas_stds('rpm')
plot_rolling_meas_stds('motor_voltage', window=300)
plot_rolling_meas_stds('motor_temp', window=300)


def adfuller_test(unit, feature, proportion=0.05, log=False):
    # df_unit = load_unit_data(unit)[[feature]]
    df_last_prop = get_last_proportion(unit, feature, proportion=proportion)
    if log == True:
        df_last_prop[feature] = np.log(df_last_prop[feature])

    adf_result = adfuller(df_last_prop[feature])
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])


adfuller_test(19, 'motor_voltage')
plot_feature_in_single_file("motor_voltage", 19, ylim_low=100, ylim_high=400, bins=100, show_dist=False)


def plot_last_daily_trend(unit, feature, proportion=0.01, log=False):
    df_last_prop = get_last_proportion(unit, feature, proportion=proportion)
    if log == True:
        df_last_prop[feature] = np.log(df_last_prop[feature])
    sns.set()
    fig, axs = plt.subplots(figsize=(16, 10))
    df_last_prop[feature].plot()


plot_last_daily_trend(19, 'motor_voltage', proportion=0.005, log=True)


def plot_acf_pacf(unit, feature, proportion=0.05, lags=50):
    df_last_prop = get_last_proportion(unit, feature, proportion=proportion)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df_last_prop[feature], lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df_last_prop[feature], lags=lags, ax=ax2)


plot_acf_pacf(19, "motor_voltage", proportion=0.005, lags=100)


def grid_search_SARIMAX(unit, feature, val_range=(0, 3), skip_diff=True, proportion=0.05, log=False):
    P = D = Q = range(val_range[0], val_range[1])
    if skip_diff:
        D = [0]
    PDQ = list(itertools.product(P, D, Q))
    seasonal_pdq = [(x[0], x[1], x[2], 24) for x in PDQ]
    # df_unit = load_unit_data(unit)[feature]
    if log == False:
        df_unit = get_last_proportion(unit, feature, proportion=proportion)[feature]
    else:
        df_unit = np.log(get_last_proportion(unit, feature, proportion=proportion)[feature])

    index_p85_input = int(len(df_unit.index) * 0.85)
    train_input = df_unit[:index_p85_input]
    test_input = df_unit[index_p85_input:]
    params_best, param_seasonal_best, AIC_best = None, None, float("inf")
    best_AIC = float("inf")
    for param in PDQ:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train_input, order=param, seasonal_order=param_seasonal,
                                                enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                if best_AIC > results.aic:
                    params_best, param_seasonal_best, best_AIC = param, param_seasonal, results.aic
                    print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))

            except Exception as ex:
                print(ex.message)
                continue
    print("best hyperparameters by grid search: ARIMA{}, Seasonal{}".format(params_best, param_seasonal_best))


def peek_attributes_last_trend(feature_name, proportion=0.05, ylim_low=-10, ylim_high=1500):
    units = random.sample(range(0, 20), 9)
    sns.set()
    f, axs = plt.subplots(3, 3, figsize=(20, 12))
    for idx, unit in enumerate(units):
        df_unit = get_last_proportion(unit, feature_name, proportion=proportion)
        unit_name = "000{}".format(unit) if unit < 10 else "00{}".format(unit)
        ax1 = plt.subplot(3, 3, idx + 1)
        df_unit[feature_name].plot(ylim=(ylim_low, ylim_high))
        ax1.set_title("{} {}".format(unit_name, feature_name))
    plt.show()


def train_SARIMAX(unit, feature, order, seasonal_order, proportion=0.05, log=False):
    #df_unit = get_last_proportion(unit, feature, proportion=proportion)[feature]
    if log == False:
        df_unit = get_last_proportion(unit, feature, proportion=proportion)[feature]
    else:
        df_unit = np.log(get_last_proportion(unit, feature, proportion=proportion)[feature])
    index_p85 = int(len(df_unit.index) * 0.85)
    train = df_unit[:index_p85]
    test = df_unit[index_p85:]
    model = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=-1)
    print(model_fit.summary())
#     plt.figure(figsize=(12, 5), dpi=100)
#     plt.plot(train, label='actual')
#     plt.plot(model_fit.fittedvalues[1:], label='fitted')
#     plt_title = "SARIMAX fitted vs actual on {} of unit {}".format(feature, unit)
#     plt.title(plt_title)
#     plt.legend(loc="upper left")
    return model_fit


model_motor_voltage_19 = train_SARIMAX(19, "motor_voltage", (1, 0, 2), (2, 0, 1, 24), proportion=0.005, log=True)


def SARIMAX_predict(model, unit, feature, proportion=0.05):
    #df_unit = load_unit_data(unit)[feature]
    df_unit = get_last_proportion(unit, feature, proportion=proportion)[feature]
    index_p85 = int(len(df_unit.index) * 0.85)
    #start, end = df_unit.index[index_p85], df_unit.index[-1]
    start, end = index_p85, len(df_unit.index)
    df_last_p15_index = df_unit[(index_p85 - 1):].index
    forecast = np.exp(model.predict(start=start, end=end, dynamic=True))
    df_forecast = pd.DataFrame(forecast, index=df_last_p15_index)
    forecast = pd.DataFrame(forecast.values, index=df_last_p15_index.tolist())
    #print(forecast.values.flatten())

    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 5), dpi=100)
    predict = model.get_prediction(start=start, end=end, dynamic=False)
    pred_ci = predict.conf_int()
    ax = sns.lineplot(x=df_unit[:index_p85 + 2].index, y=df_unit[:index_p85 + 2].values, label='training')
    ax.fill_between(forecast.index, np.exp(pred_ci.iloc[:, 0]), np.exp(pred_ci.iloc[:, 1]), color='k', alpha=.3)
    ax.fill_betweenx(ax.get_ylim(), forecast.index[0], forecast.index[-1], alpha=.2, zorder=-1)
    sns.lineplot(x=forecast.index, y=forecast.values.flatten(), label='forecast')
    plt_title = "prediction by SARIMAX on {} of unit {}".format(feature, unit)
    plt.title(plt_title)
    plt.legend(loc='upper left')


SARIMAX_predict(model_motor_voltage_19, 19, "motor_voltage", proportion=0.005)
grid_search_SARIMAX(19, 'motor_voltage', val_range=(0, 3), skip_diff=True, proportion=0.05, log=True)
plot_last_daily_trend(18, 'motor_temp', proportion=0.004, log=True)

plot_acf_pacf(18, "motor_temp", proportion=0.004, lags=100)
adfuller_test(18, 'motor_temp', 0.04, log=True)
grid_search_SARIMAX(18, 'motor_temp', val_range=(0, 3), skip_diff=True, proportion=0.04, log=True)

model_motor_temp_18 = train_SARIMAX(18, "motor_temp", (2, 0, 1), (5, 0, 1, 24), proportion=0.004, log=True)
SARIMAX_predict(model_motor_temp_18, 18, "motor_temp", proportion=0.004)





