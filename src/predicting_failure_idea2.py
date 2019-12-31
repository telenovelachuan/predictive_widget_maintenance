import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, BatchNormalization, PReLU, Dropout
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load
from datetime import timedelta, datetime
import warnings
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
KMEANS_CLUSTER1_TRAIN = [0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 16, 17, 18, 19]
KMEANS_CLUSTER2_TRAIN = [2, 8, 9, 14, 15]
KMEANS_CLUSTER1_TEST = [20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
KMEANS_CLUSTER2_TEST = [23]
Y_LABEL = 'days_to_failure'


def load_unit_data(unit, folder):
    unit_name = "000{}".format(unit) if unit < 10 else "00{}".format(unit)
    return pd.read_csv("../data/processed/{}/unit{}_rms_anomaly_excluded.csv".format(folder, unit_name), index_col=0)


def load_all_data(units='all'):
    if units == 'all':
        units = range(0, 20)
    df_all = pd.DataFrame()
    for idx, unit in enumerate(units):
        file_name = "../data/processed/train/unit{}_rms_more_features.csv".format("000{}".format(unit) if unit < 10 else "00{}".format(unit))
        file_df = pd.read_csv(file_name, index_col=0)
        file_df['timestamp'] = pd.to_datetime(file_df['timestamp'])
        df_all = df_all.append(file_df)
    return df_all


def get_train_test_sets(df_input, test_size=0.2):
    df_input_features = df_input[df_input.columns[1:]]
    num_pipeline = Pipeline([('std_scaler', StandardScaler())])

    train_set, test_set = train_test_split(df_input_features, test_size=test_size, random_state=42)
    x_train = num_pipeline.fit_transform(train_set.drop(columns=[Y_LABEL], axis=1))
    x_test = num_pipeline.fit_transform(test_set.drop(columns=[Y_LABEL], axis=1))
    y_train, y_test = num_pipeline.fit_transform(train_set[[Y_LABEL]]), num_pipeline.fit_transform(test_set[[Y_LABEL]])
    return x_train, x_test, y_train, y_test


df_c1 = load_all_data(units=KMEANS_CLUSTER1_TRAIN)
x_c1_train, x_c1_test, y_c1_train, y_c1_test = get_train_test_sets(df_c1)


def save_keras_model(model, history, file_name):
    model_json = model.to_json()
    with open("../models/failure_predicting/{}.json".format(file_name), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights("../models/failure_predicting/{}.h5".format(file_name))
    with open('../models/failure_predicting/{}.history'.format(file_name), 'wb') as file_history:
        pickle.dump(history.history, file_history)
    print("model saved")


def load_nn_model(model_name):
    with open('../models/failure_predicting/{}.json'.format(model_name)) as f:
        nn_model = model_from_json(f.read())
    nn_model.load_weights('../models/failure_predicting/{}.h5'.format(model_name))
    return nn_model


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def create_keras_model_c1(optimizer='adam', neuron=50, init='lecun_normal', act_alpha=0.01):
    model_sequences = [
        keras.layers.Dense(units=neuron, kernel_initializer=init, input_shape=x_c1_train.shape[1:]),
        keras.layers.LeakyReLU(alpha=act_alpha)
    ]

    for i in range(5):
        model_sequences.append(keras.layers.Dense(units=neuron, kernel_initializer=init,
                                                  input_shape=x_c1_train.shape[1:]))
        model_sequences.append(keras.layers.LeakyReLU(alpha=act_alpha))

    model_sequences.append(Dense(units=1, name='score_output'))
    model_sequences.append(keras.layers.LeakyReLU(alpha=act_alpha))
    nn_model = keras.models.Sequential(model_sequences)

    nn_model.compile(optimizer=optimizer, loss='mse', metrics=[coeff_determination])
    return nn_model


nn_model = create_keras_model_c1(optimizer='Adadelta', neuron=50, init='lecun_uniform', act_alpha=0.05)
print("Keras model constructed")
early_stopping_cb = EarlyStopping(patience=20)
history = nn_model.fit(x_c1_train, y_c1_train, validation_data=(x_c1_test, y_c1_test), epochs=250, callbacks=[early_stopping_cb], verbose=1)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

nn_model_c1 = nn_model
save_keras_model(nn_model_c1, history, "c1_300_epochs")

df_c2 = load_all_data(units=KMEANS_CLUSTER2_TRAIN)
x_c2_train, x_c2_test, y_c2_train, y_c2_test = get_train_test_sets(df_c2)


def create_keras_model_c2(optimizer='adam', neuron=50, init='lecun_normal', act_alpha=0.01):
    model_sequences = [
        keras.layers.Dense(units=neuron, kernel_initializer=init, input_shape=x_c2_train.shape[1:]),
        # keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=act_alpha),
    ]

    for i in range(3):
        model_sequences.append(keras.layers.Dense(units=neuron, kernel_initializer=init,
                                                  input_shape=x_c2_train.shape[1:],
                                                  # kernel_regularizer=keras.regularizers.l2(0.01)
                                                  ))
        # model_sequences.append(keras.layers.BatchNormalization())
        model_sequences.append(keras.layers.LeakyReLU(alpha=act_alpha))
        # model_sequences.append(keras.layers.PReLU())

    model_sequences.append(keras.layers.Dense(units=1, name='score_output'))
    model_sequences.append(keras.layers.LeakyReLU(alpha=act_alpha))

    nn_model = keras.models.Sequential(model_sequences)

    nn_model.compile(optimizer=optimizer, loss='mse', metrics=[coeff_determination])
    return nn_model


adadelta = keras.optimizers.Adadelta(lr=1e-3)
nn_model = create_keras_model_c2(optimizer=adadelta, neuron=50, init='lecun_normal', act_alpha=0.05)
# checkpoint_cb = ModelCheckpoint("../models/failure_predicting/c2_checkpoint.h5", save_best_only=True)
print("Keras model constructed")
history = nn_model.fit(x_c2_train, y_c2_train, validation_data=(x_c2_test, y_c2_test), epochs=300, callbacks=[], verbose=1)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

nn_model = create_keras_model_c2(optimizer=adadelta, neuron=50, init='lecun_normal', act_alpha=0.05)
early_stopping_cb = EarlyStopping(patience=10)
print("Keras model constructed")
history = nn_model.fit(x_c2_train, y_c2_train, validation_data=(x_c2_test, y_c2_test), epochs=300, callbacks=[early_stopping_cb], verbose=1)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

save_keras_model(nn_model, history, "c2_300_epochs")

rf_c2 = RandomForestRegressor(n_estimators=400, verbose=1, bootstrap=True, criterion='mse')
rf_c2.fit(x_c2_train, y_c2_train)
print("random forest trained, saving model...")
dump(rf_c2, '../models/failure_predicting/c2_rf.joblib')
print("saving model finished, getting validation scores...")
scores = cross_val_score(rf_c2, x_c2_train, y_c2_train, cv=3, scoring='r2')
print("cross val scores:{}".format(scores))


def get_time_delta_seconds(time_delta):
    return time_delta.seconds + time_delta.days * 86400


def predict_unit(unit, delta_days=15):
    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    unit_name = "000{}".format(unit) if unit < 10 else "00{}".format(unit)
    df_pred_unit_original = pd.read_csv("../data/processed/test/unit{}_rms_more_features.csv".format(unit_name), index_col=0)
    nd_pred_unit = num_pipeline.fit_transform(df_pred_unit_original[df_pred_unit_original.columns[1:]])
    model = load_nn_model("c1_300_epochs") if unit in KMEANS_CLUSTER1_TEST else load_nn_model("c2_300_epochs")
    predictions_scaled = model.predict(nd_pred_unit)

    df_train = df_c1 if unit in KMEANS_CLUSTER1_TEST else df_c2
    y_scaler = StandardScaler().fit(df_train[[Y_LABEL]])
    predictions = y_scaler.inverse_transform(predictions_scaled)

    every_fail_time = []
    df_pred_unit_original['timestamp'] = pd.to_datetime(df_pred_unit_original['timestamp'])
    date_max = max(df_pred_unit_original['timestamp'])
    predicted_failures = 0
    for num_idx, pred in enumerate(predictions):
        pred = float(predictions[num_idx][0])
        try:
            failure_time = df_pred_unit_original.iloc[num_idx]['timestamp'] + timedelta(days=pred)
        except Exception as ex:
            print(ex)
            print("!!pred:" + str(pred))
            continue
            # failure_time = df_pred_unit_original.iloc[num_idx]['timestamp'] + longest_span

        # predicted to fail within delta_days
        if failure_time <= (date_max + timedelta(days=delta_days)):
            predicted_failures += 1
        every_fail_time.append(failure_time)

    date_min, date_max = min(every_fail_time), max(every_fail_time)
    # print('date_min:{}, date_max:{}'.format(date_min, date_max))
    time_deltas = [get_time_delta_seconds(spot - date_min) for spot in every_fail_time]
    avg_time_delta = sum(time_deltas) / len(time_deltas)
    # print("avg_time_delta:{}".format(avg_time_delta))
    final_avg_result = date_min + timedelta(seconds=avg_time_delta)
    # print("unit {} final avg result:{}".format(unit, final_avg_result))
    print("unit {} failure prob:{}, avg:{}".format(unit, float(predicted_failures) / len(predictions), final_avg_result))
    return final_avg_result


avg_results = [predict_unit(unit, delta_days=30) for unit in range(20, 50)]


