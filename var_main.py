# Please refer to the var_preprocessing file to see the insights gained from the preprocessing steps
# Generalised order is 7

import src.constants as const
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from sklearn import preprocessing
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
import src.evaluation as eval


def adjust(val, length=6):
    return str(val).ljust(length)


# Anomaly Detection code
def find_anomalies(squared_errors, z=1):
    threshold = np.mean(squared_errors) + (z * np.std(squared_errors))
    predictions = (squared_errors >= threshold).astype(int)
    return predictions, threshold


if __name__ == '__main__':
    # Pre-requisites
    min_max_scaler = preprocessing.MinMaxScaler()

    dataset_path = const.PROPRIETARY_DATASET_LOCATION
    # Read anomaly data
    anomaly_path = join(dataset_path, 'anomaly_data/')
    # As the first step, combine the csvs inside anomaly_data folder
    anomaly_data_files = [f for f in listdir(anomaly_path) if isfile(join(anomaly_path, f))]
    anomaly_df_list = [pd.read_csv(anomaly_path + anomaly_data_file) for anomaly_data_file in anomaly_data_files]
    # Next, drop the datetime column
    anomaly_df_list_without_datetime = [anomaly_df.drop(columns=['datetime']) for anomaly_df in anomaly_df_list]
    # Finally merge those dataframes
    anomaly_df = pd.concat(anomaly_df_list_without_datetime)
    anomaly_df = anomaly_df.astype(float)
    # Separate out the is_anomaly labels before normalisation/standardization
    anomaly_df_labels = anomaly_df['is_anomaly']
    anomaly_df = anomaly_df.drop(['is_anomaly'], axis=1)

    # Initialise the VAR model by providing anomaly_df
    var_model = VAR(anomaly_df)
    # Train the VAR model of selected order
    var_model_fitted = var_model.fit(7)
    print(var_model_fitted.summary())
    # Check for Serial Correlation of Residuals (Errors) using Durbin Watson Statistic
    out = durbin_watson(var_model_fitted.resid)
    for col, val in zip(anomaly_df.columns, out):
        print(adjust(col), ':', round(val, 2))

    squared_errors = var_model_fitted.resid.sum(axis=1) ** 2
    predictions, threshold = find_anomalies(squared_errors)  # threshold = 2009202927.4907029
    y_pred = predictions.values
    y_test = anomaly_df_labels.iloc[7:]
    eval.evaluation(y_test, y_pred)
