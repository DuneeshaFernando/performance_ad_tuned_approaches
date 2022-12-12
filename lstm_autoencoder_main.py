import src.constants as const
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn import preprocessing
import config.config as config
import numpy as np
import torch
import torch.utils.data as data_utils
import src.lstm_autoencoder as lstm_autoencoder
import src.evaluation as eval

if __name__ == '__main__':
    # Pre-requisites
    min_max_scaler = preprocessing.MinMaxScaler()

    dataset_path = const.PROPRIETARY_DATASET_LOCATION
    # Read normal data
    normal_path = join(dataset_path,'normal_data/')
    # As the first step, combine the csvs inside normal_data folder
    normal_data_files = [f for f in listdir(normal_path) if isfile(join(normal_path, f))]
    normal_df_list = [pd.read_csv(normal_path + normal_data_file) for normal_data_file in normal_data_files]
    # Next, drop the datetime column
    normal_df_list_without_datetime = [normal_df.drop(columns=['datetime']) for normal_df in normal_df_list]
    # Finally merge those dataframes
    normal_df = pd.concat(normal_df_list_without_datetime)
    normal_df = normal_df.astype(float)
    # Normalise/ standardize the normal dataframe
    normal_df_values = normal_df.values
    normal_df_values_scaled = min_max_scaler.fit_transform(normal_df_values)
    normal_df_scaled = pd.DataFrame(normal_df_values_scaled) # shape = (9652, 9)

    # Read anomaly data
    anomaly_path = join(dataset_path,'anomaly_data/')
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
    # Normalise/ standardize the anomaly dataframe
    anomaly_df_values = anomaly_df.values
    anomaly_df_values_scaled = min_max_scaler.transform(anomaly_df_values)
    anomaly_df_scaled = pd.DataFrame(anomaly_df_values_scaled) # shape = (3635, 9)

    # Preparing the datasets for training and testing using AutoEncoder
    windows_normal = normal_df_scaled.values[np.arange(config.WINDOW_SIZE)[None, :] + np.arange(normal_df_scaled.shape[0] - config.WINDOW_SIZE)[:, None]] # shape = (9647, 5, 9)
    windows_anomaly = anomaly_df_scaled.values[np.arange(config.WINDOW_SIZE)[None, :] + np.arange(anomaly_df_scaled.shape[0] - config.WINDOW_SIZE)[:, None]] # shape = (3630, 5, 9)

    # Create batches of training and testing data
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal).float()
    ), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_anomaly).float()
    ), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialise the LSTMAutoEncoder model
    lstm_autoencoder_model = lstm_autoencoder.LstmAutoencoder(seq_len=config.WINDOW_SIZE, n_features=windows_normal.shape[2], num_layers=config.NUM_LAYERS)
    # Start training
    history = lstm_autoencoder.training(config.N_EPOCHS, lstm_autoencoder_model, train_loader, config.LEARNING_RATE)

    # Save the model and load the model
    model_path = const.MODEL_LOCATION
    torch.save({
        'encoder': lstm_autoencoder_model.encoder.state_dict(),
        'decoder': lstm_autoencoder_model.decoder.state_dict()
    }, join(model_path,"lstm_ae_model.pth"))
    checkpoint = torch.load(join(model_path,"lstm_ae_model.pth"))
    lstm_autoencoder_model.encoder.load_state_dict(checkpoint['encoder'])
    lstm_autoencoder_model.decoder.load_state_dict(checkpoint['decoder'])

    # Use the trained model to obtain predictions for the test set
    results = lstm_autoencoder.testing(lstm_autoencoder_model, test_loader)
    y_pred_for_test_set = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(), results[-1].flatten().detach().cpu().numpy()])

    # Obtain threshold based on pth percentile of the mean squared error
    threshold = np.percentile(y_pred_for_test_set, [90])[0]  # 90th percentile
    # Map the predictions to anomaly labels after applying the threshold
    predicted_labels = []
    for val in y_pred_for_test_set:
        if val > threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    # Evaluate the predicted_labels against the actual labels
    eval.evaluation(anomaly_df_labels[config.WINDOW_SIZE:], predicted_labels)