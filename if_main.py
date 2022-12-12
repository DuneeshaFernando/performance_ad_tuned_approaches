import src.constants as const
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.ensemble import IsolationForest
import src.evaluation as eval
from sklearn import preprocessing
import config.config as config
import numpy as np

def average(lst):
    return sum(lst) / len(lst)

def kpoint(df, k):
    kpoint_column_titles = list(df.columns.values)
    for column_name in kpoint_column_titles:
        new_column_values = []
        for j in range(len(df[column_name])):
            range_start = max(0, j-k)
            range_end = max(0, j-1)
            if range_start == range_end == 0:
                value_range = df.loc[0,column_name]
                new_column_values.append(value_range)
            else:
                value_range = df.loc[range_start:range_end,column_name]
                new_column_values.append(average(value_range))
        df[column_name+"_kpoint"] = new_column_values
    return df

if __name__ == '__main__':
    # Pre-requisites
    min_max_scaler = preprocessing.MinMaxScaler()

    dataset_path = const.PROPRIETARY_DATASET_LOCATION
    # Read normal data
    normal_path = join(dataset_path, 'normal_data/')
    # As the first step, combine the csvs inside normal_data folder
    normal_data_files = [f for f in listdir(normal_path) if isfile(join(normal_path, f))]
    normal_df_list = [pd.read_csv(normal_path + normal_data_file) for normal_data_file in normal_data_files]
    # Next, drop the datetime column
    normal_df_list_without_datetime = [normal_df.drop(columns=['datetime']) for normal_df in normal_df_list]
    # Finally merge those dataframes
    normal_df = pd.concat(normal_df_list_without_datetime) # shape = (9652, 9)
    # Add a new column called is_anomaly with the value 0 to the normal_df
    normal_df['is_anomaly'] = 0

    # Read anomaly data
    anomaly_path = join(dataset_path, 'anomaly_data/')
    # As the first step, combine the csvs inside anomaly_data folder
    anomaly_data_files = [f for f in listdir(anomaly_path) if isfile(join(anomaly_path, f))]
    anomaly_df_list = [pd.read_csv(anomaly_path + anomaly_data_file) for anomaly_data_file in anomaly_data_files]
    # Next, drop the datetime column
    anomaly_df_list_without_datetime = [anomaly_df.drop(columns=['datetime']) for anomaly_df in anomaly_df_list]
    # Finally merge those dataframes
    anomaly_df = pd.concat(anomaly_df_list_without_datetime) # shape = (3635, 10) with the is_anomaly column

    # Create the train set and test set from the anomaly_df and normal_df
    # Split the anomaly_df into 2 parts
    anomaly_df_split_point  = int(anomaly_df.shape[0] / 2)
    anomaly_df_part_1 = anomaly_df.iloc[:anomaly_df_split_point, :]
    anomaly_df_part_2 = anomaly_df.iloc[anomaly_df_split_point:, :]

    # Split the normal_df also such that anomaly_df_part_1 could replace a part of normal_df
    normal_df_split_point = normal_df.shape[0]-anomaly_df_split_point
    normal_df_part_1 = normal_df.iloc[:normal_df_split_point, :]
    normal_df_part_2 = normal_df.iloc[normal_df_split_point:, :]

    # Merge the split dataframes to form the train set and test set accordingly
    train_frames = [normal_df_part_1, anomaly_df_part_1]
    test_frames = [normal_df_part_2, anomaly_df_part_2]
    X_train = pd.concat(train_frames)
    X_test = pd.concat(test_frames)

    # Separate out the is_anomaly labels before normalisation/standardization
    y_train = X_train['is_anomaly']
    X_train = X_train.drop(['is_anomaly'], axis=1)
    y_test = X_test['is_anomaly']
    X_test = X_test.drop(['is_anomaly'], axis=1)

    # Normalise/ standardize the merged dataframe
    X_train_values = X_train.values
    X_train_values_scaled = min_max_scaler.fit_transform(X_train_values)
    X_train_scaled = pd.DataFrame(X_train_values_scaled, columns=['in_avg_response_time','in_throughput','in_progress_requests','http_error_count','ballerina_error_count','cpu','memory','cpuPercentage','memoryPercentage'])
    X_test_values = X_test.values
    X_test_values_scaled = min_max_scaler.transform(X_test_values)
    X_test_scaled = pd.DataFrame(X_test_values_scaled, columns=['in_avg_response_time','in_throughput','in_progress_requests','http_error_count','ballerina_error_count','cpu','memory','cpuPercentage','memoryPercentage'])

    # Add k-point moving average new feature
    X_train_scaled = kpoint(X_train_scaled, config.k)
    X_test_scaled = kpoint(X_test_scaled, config.k)

    # Initialise the Isolation Forest model with the best hyper-parameters and train it using the train set
    if_model = IsolationForest(max_features=0.6, n_estimators=330, max_samples=0.1).fit(X_train_scaled)

    X_test_scaled['y_pred'] = if_model.score_samples(X_test_scaled)
    threshold = np.percentile(X_test_scaled['y_pred'], [9.13])[0]

    predicted_labels = []
    X_test_scaled['is_anomaly'] = X_test_scaled['y_pred'] < threshold
    # predicted_labels = X_test_scaled['y_pred'].map({1: 0, -1: 1})

    eval.evaluation(y_test,X_test_scaled['is_anomaly'])
