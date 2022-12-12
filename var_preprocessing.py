# Unlike other ML approaches, VAR is a statistical approach. Finding the order is the most critical step in the VAR approach.
# Therefore, in this file, I will run some initial tests and determine the most common order for individual files.

import src.constants as const
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn import preprocessing
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-' * 47)
    # print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    # print(f' Significance Level    = {signif}')
    # print(f' Test Statistic        = {output["test_statistic"]}')
    # print(f' No. Lags Chosen       = {output["n_lags"]}')
    #
    # for key,val in r[4].items():
    #     print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


if __name__ == '__main__':
    # Pre-requisites
    min_max_scaler = preprocessing.MinMaxScaler()
    maxlag = 12
    test = 'ssr_chi2test'

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

    # Conduct tests on merged dataset
    # Conduct Granger's Causality Test
    grangers_causation_matrix = grangers_causation_matrix(anomaly_df, variables=anomaly_df.columns)
    # Conduct a cointegration test
    cointegration_test(anomaly_df)
    # Make the time-series stationary. This time-series becomes stationery at the first time itself
    for name, column in anomaly_df.iteritems():
        adfuller_test(column, name=column.name)
        print('\n')
    # Find the generalised order p, i.e. the common order for all datasets
    var_model = VAR(anomaly_df)
    var_order_selection_matrix = var_model.select_order(maxlags=12)
    print(var_order_selection_matrix.summary())  # Generalised order is p=7

    # Find the order for each individual dataset. But obtaining order at individual datafile level returns the error "n-th leading minor of the array is not positive definite"
    for individual_anomaly_df in anomaly_df_list_without_datetime:
        # Remove all-zero columns before obtaining the order
        individual_anomaly_df = individual_anomaly_df.loc[:, (individual_anomaly_df != 0).any(axis=0)]
        individual_anomaly_df = individual_anomaly_df.drop(['is_anomaly'], axis=1)
        # print(individual_anomaly_df.columns) # This is to test the remaining data columns after all-zero columns are removed
        try:
            individual_var_model = VAR(individual_anomaly_df)
            individual_var_order_selection_matrix = individual_var_model.select_order(maxlags=12)
            print(individual_var_order_selection_matrix.summary())
        except:
            print("n-th leading minor of the array is not positive definite")
