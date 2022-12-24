import pandas as pd
from scipy.fft import fft


def generate_features(path: str, sample_rate: str = '2857143ns', original: bool = False) -> pd.DataFrame:
    """
    This function generate multiple features from given csv and returns a DataFrame
    :param original: if True only return the original features
    :param path: Path of the csv file
    :param sample_rate: sample rate to down sample the DF
    :return df_transformed:
    """

    # read the data, remove the unnecessary column and set index to time_stamps
    df = pd.read_csv(path)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.set_index(pd.DatetimeIndex(df['time_stamps']), inplace=True)
    df.dropna(inplace=True)
    df.drop(['time_stamps'], axis=1, inplace=True)

    # Get the column names
    columns = [f for f in df.columns if f not in ['label', 'time_stamps']]

    # Resample at given HZ
    df = df.resample(sample_rate).ffill()
    if not original:
        # Create Fourier transformations
        # for col in columns:
        #     df[f'{col}_fourier'] = fft(df[col].values)
        # # New column names after fft
        # columns = [f for f in df.columns if f not in ['label', 'time_stamps']]
        # create polynomial features #
        # # initialize pf object
        # pf = preprocessing.PolynomialFeatures(degree=2,
        #                                       interaction_only=False,
        #                                       include_bias=False)
        #
        # # TODO Accelerometer signals might be removed
        # # fit to the features
        # pf.fit(df.iloc[:, :-1])
        #

        # # create polynomial features
        # poly_feats = pf.transform(df.iloc[:, :-1])
        #
        # # create a dataframe with new features
        # num_feats = poly_feats.shape[-1]
        # df_transformed_poly = pd.DataFrame(poly_feats, columns=pf.get_feature_names_out())

        # Create Binning Features #
        # Initialize a DF
        df_binning_features = pd.DataFrame()
        for col in columns:
            df_binning_features[f'{col}_bin_10'] = pd.cut(df[col], bins=10, labels=False)

        # Create Lag Features #
        # list of lags
        lags = [100, 350, 700, 3500, 7000]
        # new df to store Lag Features
        df_lag_features = pd.DataFrame()
        # for every element in lags create lag features
        for lag in lags:
            df[f'lag_{lag}'] = df['label'].shift(lag)

        # Create Rolling Features
        # list of window sizes
        windows = [50, 100, 350, 700, 3500, 7000]
        # Create df to store rolling features
        df_rolling_features = pd.DataFrame()
        # for every column create rolling features in given window size
        for col in columns:
            for win in windows:
                df_rolling_features[f'{col}_rolling_{win}_mean'] = df[col].rolling(win).mean()
                df_rolling_features[f'{col}_rolling_{win}_max'] = df[col].rolling(win).max()
                df_rolling_features[f'{col}_rolling_{win}_min'] = df[col].rolling(win).min()
        # Create Expanding Features #

        df_expanding_features = pd.DataFrame()
        for col in columns:
            df_expanding_features[f'{col}_expanding'] = df[col].expanding(min_periods=2).obj

        # Concat all new features into single df
        df_transformed = pd.concat(
            [df[columns], df_lag_features, df_rolling_features,
             df_expanding_features, df.label], axis=1)

        mask = df_transformed['label'].isin([0, 4, 5, 6, 7])
        df_transformed = df_transformed[~mask]

        return df_transformed

    else:
        mask = df['label'].isin([0, 4, 5, 6, 7])
        df = df[~mask]
        return df
