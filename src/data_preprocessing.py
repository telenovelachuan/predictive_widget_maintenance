import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def peek_attributes_trend(attribute, units):
    file_names = ["../data/raw/train/unit{}_rms.csv".format(unit) for unit in units]
    sns.set()
    f, axs = plt.subplots(2, 2, figsize=(15, 7))
    for idx, file_name in enumerate(file_names):
        file_df = pd.read_csv(file_name)
        ax1 = plt.subplot(2, 2, idx + 1)
        file_df[attribute].plot(ylim=(-10, 1500))
    plt.show()


peek_attributes_trend('motor_voltage', ['0005', '0012', '0007', '0019'])

