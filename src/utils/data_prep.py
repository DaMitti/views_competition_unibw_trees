"""
Collection of utility functions used to load and/or prep data to be used in the prediction pipeline. Mainly developed
for the priogrid-month level. Functionality for country-months is only partially implemented at this time.
"""

import os
from typing import Literal, Iterable

import numpy as np
import pandas as pd
import pendulum
from sklearn.datasets import make_regression
import xarray as xr

from src.utils.conversion import get_month_id


def get_shifted_values(df: pd.DataFrame, feature: str, timeshifts: Iterable[int|float], loc_index: str) -> pd.DataFrame:
    """
    Shifts features by 'timeshifts', taking into account different locations.
    
    Args:
        df: input dataframe which contains the feature to shift and the location information.
        feature: column name of the feature to shift
        timeshifts: List of timeshift values for which to create shifted versions. For each entry, positive values 
            result in future values and negative ones in lagged values, e.g. input 1 results in t+1 and input -1 in t-1 
            version of the respective feature.
        loc_index: string with name of the column that contains location information 

    Returns:
        DataFrame with shifted features with the index of 'df'.
    """
    temp_df = df.copy()
    temp_df = temp_df.reset_index().groupby(loc_index)[feature]
    new_features = {}
    for timeshift in timeshifts:
        val = temp_df.shift(timeshift * -1)
        new_features[f'{feature}_t{timeshift}'] = val
        new_features[f'dummy_{feature}_t{timeshift}'] = [1 if v > 0 else v for v in val]

    return pd.DataFrame(new_features).set_index(df.index)


def level_index(level: str) -> list[str]:
    """ small helper function to return the appropriate index for the prediction level """
    if level == 'pgm':
        index = ['month_id', 'priogrid_gid']
    elif level == 'cm':
        index = ['month_id', 'country_id']
    else:
        raise ValueError(f"Argument level needs to be one of ['cm', 'pgm'], got {level}")
    return index


def read_prio_pgm_features(fp: str, prediction_year: int|pendulum.Date = 2018, start_year: int = 1990,
                           level: Literal['cm', 'pgm'] = 'pgm') -> pd.DataFrame:
    """
    Loads prio features cumulative from the start year up to three months before the prediction window specified. This
    means e.g. data up to and including Oct 2020 for prediction_year=2021.

    NOTE: The 2024 window corresponds to July 2024 to June 2025 in accordance with the ViEWS challenge while other
    windows reflect Jan-Dec of the specified year.

    Args:
        fp: filepath to ViEWS data folder
        prediction_year: year of the prediction window for which to load the features
        start_year: first year to load features from
        level: 'cm' or 'pgm'
    """
    if prediction_year == 2024:
        features_end_year = 2025
    else:
        features_end_year = prediction_year
    fp_features = os.path.join(fp, f'features/{level}/')
    if level == 'cm':
        df_f = pd.read_parquet(os.path.join(fp_features, 'cm_features.parquet')).set_index(level_index(level))
        assert df_f.index.names == level_index(level) # integrity check
    elif level == 'pgm':
        assert start_year >= 1990 # data only goes back to 1990
        dfs = []
        for year in range(start_year, features_end_year):
            fp_year = os.path.join(fp_features, f'year={year}')
            filename = os.listdir(fp_year)[0]
            df_temp = pd.read_parquet(os.path.join(fp_year, filename)).set_index(level_index(level))
            try:
                assert df_temp.index.names == level_index(level)  # integrity check
            except AssertionError:
                print('Index check failed at year', year)
                raise
            dfs.append(df_temp)
        df_f = pd.concat(dfs, axis=0)
        df_f.ged_sb = df_f.ged_sb.apply(lambda x: round(x, 0)) # correct floating point precision errors
    else:
        raise ValueError(f"Argument level needs to be one of ['cm', 'pgm'], got {level}")
    # adjust end date for data
    prediction_start = pendulum.date(year=prediction_year, month=1, day=1)
    # handle the prediction challenge's true future prediction window from July '24 to June '25
    if prediction_year == 2024:
        prediction_start = pendulum.date(year=prediction_year, month=7, day=1)
    last_month = get_month_id(prediction_start.subtract(months=3))
    df_f = df_f.loc[:last_month]
    return df_f


def read_prio_actuals(fp: str, year: int, as_xarray: bool = True,
                      level: Literal['cm', 'pgm'] = 'pgm') -> xr.DataArray|pd.DataFrame:
    """
    Reads the actuals provided by ViEWS for a given prediction window.

    NOTE: Actuals for 2024 only include data up to April.

    Args:
        fp: filepath to ViEWS data folder
        year: year of the prediction window for which to read actuals.
        as_xarray: whether to return the data as DataArray or as DataFrame.
        level: prediction resolution (either "cm" for country-month or "pgm" for priogrid-month).

    Returns:
        DataArray or DataFrame with actuals for the year depending on as_xarray=True. Data will be returned in
        accordance with the evaluation naming scheme as "outcome" for DataArrays, and as "ged_sb" for DataFrames for
        compatibility with feature naming.
    """
    # simple argument check
    assert year in range(2018, 2024+1)
    fp_actuals = os.path.join(fp, f'actuals/{level}/window=Y{year}')
    filename = f'{level}_actuals_{year}.parquet'
    df = pd.read_parquet(os.path.join(fp_actuals, filename))
    assert df.index.names == level_index(level)

    df.outcome = df.outcome.astype(int) # correct potential floating point precision errors and reduce size
    if as_xarray:
        da = df.outcome.to_xarray()
        return da
    else:
        # dataframes are used where we are merging with data from features, so we use ged_sb for consistency
        return df.rename(columns={'outcome':'ged_sb'})


def create_dummy_data(prediction_year: int) -> pd.DataFrame:
    """
    Creates dummy regression dataset with 10 features including target for 100 pgids and ~11 years for pipeline testing.
    """
    index_names = level_index('pgm')
    priogrid_gids = np.arange(100)
    prediction_start = pendulum.date(year=prediction_year, month=1, day=1)
    month_ids = np.arange(get_month_id(prediction_start.subtract(years=11)),
                          get_month_id(prediction_start.subtract(months=2)))
    index = pd.MultiIndex.from_product([month_ids, priogrid_gids], names=index_names)
    X, y = make_regression(n_samples=len(index), n_features=9, n_informative=4, noise=.5)
    column_names = [f'feature{i}' for i in range(9)]
    df_dummy = pd.DataFrame(data=X, index=index, columns=column_names)
    df_dummy['ged_sb'] = y/100
    df_dummy = df_dummy.abs().round(0)
    df_dummy = df_dummy.sort_index(axis=1, ascending=False)
    return df_dummy


def read_prio_training_data(fp:str, prediction_year:int=2018,
                            test_data: bool = False) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Reads the required training data for a given prediction window. Reads features and generates the corresponding
    prediction targets by shifting for t+3 to t+14 (features end 3 months before the start of a prediction window)

    NOTE: The 2024 window corresponds to July 2024 to June 2025 in accordance with the ViEWS challenge while other
    windows reflect Jan-Dec of the specified year.

    Args:
        fp: filepath to ViEWS data folder
        prediction_year: year of the prediction window
        test_data: if true, create smaller dummy training datasets for testing of the pipeline rather than loading the
            actual data

    Returns:
        DataFrame with training data and a dictionary with lists of features, regression targets and classification
        targets.
    """
    # simple argument check
    assert prediction_year in range(2018, 2025 + 1) # this can go to 2025 for the true predictions

    if not test_data:
        df_f = read_prio_pgm_features(fp, prediction_year)
    else:
        df_f = create_dummy_data(prediction_year)

    shifts = np.arange(3, 15)
    df_t = get_shifted_values(df_f, 'ged_sb', shifts, 'priogrid_gid')
    f_dict = {
        'features': list(df_f.columns)[:-1],
        'targets_reg': [t for t in df_t.columns if 'dummy' not in t],
        'targets_clf': [t for t in df_t.columns if 'dummy' in t]
    }

    df_train = pd.concat([df_f, df_t], axis=1, join='inner')
    df_train = df_train.sort_index()

    return df_train, f_dict


def read_predictions(fp:str, year:int, level: Literal['cm', 'pgm'] = 'pgm',
                     as_xarray: bool = True) -> pd.DataFrame|xr.DataArray:
    """
    Function to read the generated predictions for a given prediction window.

    NOTE: The 2024 window corresponds to July 2024 to June 2025 in accordance with the ViEWS challenge while other
    windows reflect Jan-Dec of the specified year.

    Args:
        fp: filepath to submission folder where predictions are stored
        level: prediction resolution (either "cm" for country-month or "pgm" for priogrid-month).
        year: year of the prediction window
        as_xarray: whether to return the data as DataArray or as DataFrame.

    Returns:
        DataFrame or DataArray with predictions.
    """
    # needs 18ish GB RAM for 1 year
    assert year in range(2018, 2024+1)
    fp_window = os.path.join(fp, level, f'window=Y{year}')
    predictions = pd.read_parquet(os.path.join(fp_window, f'{fp}_predictions_{year}.parquet'))
    assert predictions.index.names == level_index(level) + ['draw']

    # default is retuning as xarray
    if as_xarray:
        da = predictions.outcome.to_xarray()
        return da

    else:
        # just the output of pandas read
        return predictions


def read_benchmarks(fp: str, year: int, benchmark: str, level: Literal['cm', 'pgm'] = 'pgm',
                    as_xarray: bool = True) -> xr.DataArray|pd.DataFrame:
    """
    Function to read the provided benchmark "predictions" for a given prediction window.

    NOTE: The 2024 window corresponds to July 2024 to June 2025 in accordance with the ViEWS challenge while other
    windows reflect Jan-Dec of the specified year.

    Args:
        fp: filepath to submission folder where predictions are stored
        level: prediction resolution (either "cm" for country-month or "pgm" for priogrid-month).
        year: year of the prediction window
        as_xarray: whether to return the data as DataArray or as DataFrame.

    Returns:
        DataFrame or DataArray with predictions.
    """
    # simple argument checks
    assert year in range(2018, 2024+1)
    assert benchmark in ['conflictology_n', 'last', 'zero', 'conflictology', 'boot_240']

    fp_benchmark = os.path.join(fp, f'benchmarks/{benchmark}/{level}/window=Y{year}')
    filename = f'bm_{benchmark}_{level}_{year}.parquet'
    df = pd.read_parquet(os.path.join(fp_benchmark, filename))
    assert df.index.names == level_index(level) + ['draw']

    if as_xarray:
        da = df.outcome.to_xarray()
        return da
    else:
        return df


def read_raw_predictions(fp_views:str, fp_pipeline:str, years:int|list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to read past "raw" predictions as defined by years argument for calibration of the hurdle ensemble.

    Args:
        fp_views: filepath to ViEWS data folder.
        fp_pipeline: filepath to yearly raw predictions from prediction pipeline.
        years: prediction windows for which to read the raw predictions, can either be single year or list of years.

    Returns:
        2-Tuple of DataFrames consisting of (raw predictions, observed values) for the specified years.
    """
    if type(years) is int:
        years = [years]

    observed = []
    predictions_raw=[]

    for year in years:
        fp_predictions = os.path.join(fp_pipeline, f'predictions_{year}.parquet')
        df_predictions_raw = pd.read_parquet(fp_predictions)
        predictions_raw.append(df_predictions_raw)

        # we only have "actuals" files from 2018, so for anything earlier we need to read fatalities from the features
        if year < 2018:
            # the function reading the features always returns the features up to Oct from last year - if we want full
            # data for 2017 to be included we therefore need to read the features for 2019 predictions
            prediction_year = 2018 if year < 2017 else 2019
            months = df_predictions_raw.index.get_level_values('month_id').unique()
            df_observed = read_prio_pgm_features(fp_views, prediction_year).loc[months, ['ged_sb']].copy()
        else:
            df_observed = read_prio_actuals(fp_views, year, as_xarray=False)
        df_observed['ged_dummy'] = df_observed['ged_sb'].apply(lambda x: int(x>0))
        observed.append(df_observed)

    predictions_raw = pd.concat(predictions_raw).sort_index()
    observed = pd.concat(observed).sort_index()

    return predictions_raw, observed


