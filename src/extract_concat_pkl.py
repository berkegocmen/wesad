import pickle
import pandas as pd
from datetime import datetime


def extract_concat_pkl(path: str = '../data/S2/S2.pkl') -> pd.DataFrame:
    """
    Extracts the data from the pickle file and concatenates it into a single dataframe.
    :param path:
    :return:
    """

    # Load the pickle file
    u = pickle.Unpickler(open(path, 'rb'), encoding='latin1')
    pickled_data = u.load()

    # Extract the data from the pickle file
    labels = pickled_data['label']
    chest = pickled_data['signal']['chest']
    wrist = pickled_data['signal']['wrist']

    # Generate timestamps in different frequencies
    time_stamps_at_700 = pd.date_range(datetime(2022, 1, 1, hour=00, minute=00), periods=len(labels), freq='1428571ns')
    time_stamps_at_32_acc = pd.date_range(datetime(2022, 1, 1, hour=00, minute=00), periods=wrist['ACC'].shape[0],
                                          freq='31.25ms')
    time_stamps_at_64_bvp = pd.date_range(datetime(2022, 1, 1, hour=00, minute=00), periods=wrist['BVP'].shape[0],
                                          freq='15.625ms')
    time_stamps_at_4_eda_temp = pd.date_range(datetime(2022, 1, 1, hour=00, minute=00), periods=wrist['TEMP'].shape[0],
                                              freq='250ms')

    # Create a dataframe from chest data
    chest_df = pd.DataFrame()
    for (key, value) in chest.items():
        if value.shape[1] > 1:
            for channel in range(value.shape[1]):
                chest_df['chest_' + key + '_' + str(channel)] = value[:, channel]
        else:
            chest_df['chest_' + key] = value[:, 0]

    chest_df['time_stamps'] = time_stamps_at_700
    chest_df.set_index('time_stamps', inplace=True)

    # Create a dataframe from wrist data
    wrist_acc = pd.DataFrame(wrist['ACC'], columns=['wrist_ACC_0', 'wrist_ACC_1', 'wrist_ACC_2'])
    wrist_acc['time_stamps'] = time_stamps_at_32_acc
    wrist_acc.set_index('time_stamps', inplace=True)

    wrist_bvp = pd.DataFrame(wrist['BVP'], columns=['wrist_BVP'])
    wrist_bvp['time_stamps'] = time_stamps_at_64_bvp
    wrist_bvp.set_index('time_stamps', inplace=True)

    wrist_eda = pd.DataFrame(wrist['EDA'], columns=['wrist_EDA'])
    wrist_eda['time_stamps'] = time_stamps_at_4_eda_temp
    wrist_eda.set_index('time_stamps', inplace=True)

    wrist_temp = pd.DataFrame(wrist['TEMP'], columns=['wrist_TEMP'])
    wrist_temp['time_stamps'] = time_stamps_at_4_eda_temp
    wrist_temp.set_index('time_stamps', inplace=True)

    # Resample the wrist data to the same frequency as the chest data
    wrist_acc = wrist_acc.resample('1428571ns').ffill()
    wrist_bvp = wrist_bvp.resample('1428571ns').ffill()
    wrist_eda.set_index('time_stamps', inplace=True)
    wrist_eda = wrist_eda.resample('1428571ns').ffill()
    wrist_temp.set_index('time_stamps', inplace=True)
    wrist_temp = wrist_temp.resample('1428571ns').ffill()

    # Concatenate the dataframes
    df = pd.concat([chest_df, wrist_acc, wrist_bvp, wrist_eda, wrist_temp], axis=1)

    # return the dataframe
    return df
