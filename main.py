# ---------------------------
# import
# ---------------------------

import pandas as panda
import numpy as np
from os import walk
from typing import Dict
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pathlib
from k_nearest_neigbours import KnnDtw
from sklearn.metrics import classification_report, confusion_matrix
import re


def load_data_file(path: str):
    return panda.read_csv(
        path,
        sep=' ',
        header=None
    )


def save_house_files(house: str) -> None:
    (_, __, file_names) = next(walk('data/low_freq/' + house + '/'))
    file_names.remove('labels.dat')

    for file_name in file_names:
        signal_dataframe = load_data_file('data/low_freq/' + house + '/' + file_name)
        signal_dataframe.columns = ['timestamp', 'power']
        signal_dataframe.set_index('timestamp', inplace=True)
        signal_dataframe.index = panda.to_datetime(signal_dataframe.index, unit='s')
        signal_series = panda.Series(signal_dataframe['power'], signal_dataframe.index)
        signal_series.to_pickle('./data/converted/' + house + '/' + re.sub("\.dat$", '', file_name) + '.pkl')


def load_house_files() -> Dict:
    (_, _, file_names) = next(walk('data/converted/'))

    files = {}
    for file_name in file_names:
        files[file_name] = panda.read_pickle('data/converted/' + file_name)

    return files


def build_original_signal(house: str):
    pickled_series_path = 'data/converted/' + house + '/'
    (_, __, file_names) = next(walk(pickled_series_path))

    first_iteration = True
    for file_name in file_names:
        if first_iteration:
            original_signal: panda.Series = panda.read_pickle(pickled_series_path + '/' + file_name)
            first_iteration = False
        else:
            single_signal = panda.read_pickle(pickled_series_path + '/' + file_name)
            original_signal.add(single_signal, fill_value=0.0)

    return original_signal


def run_fft(
        raw_signal_data: panda.DataFrame,
        number_of_samples: int = 10,
        sample_spacing: float = 0.01
) -> Dict:
    mean_subtracted_data = raw_signal_data.to_numpy()
    mean_subtracted_data = mean_subtracted_data - np.mean(mean_subtracted_data)
    yf = fft(mean_subtracted_data)
    xf = fftfreq(number_of_samples, sample_spacing)[:number_of_samples // 2]

    fft_results = {'xf': xf, 'yf': yf, }

    return fft_results


def make_fft_plot(
        fft_result: Dict,
        number_of_samples: float,
        plot_name: str,
) -> None:
    plt.plot(
        fft_result['xf'],
        2.0 / number_of_samples * np.abs(fft_result['yf'][0:number_of_samples // 2]),
        label=fft_result['name']
    )

    plt.title(plot_name)
    plt.ylabel('|DFT (K)|')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.savefig('graphs/fft/' + plot_name)
    plt.close()


def k_nearest_heighbour(raw_signal_data: Dict, labels: dict):
    # Mapping table for classes
    #labels = {1: 'WALKING', 2: 'WALKING UPSTAIRS', 3: 'WALKING DOWNSTAIRS',
    #          4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}

    x_test = raw_signal_data['x_test'].to_numpy()
    x_train = raw_signal_data['x_train'].to_numpy()

    y_test = raw_signal_data['y_test'].to_numpy()
    y_train = raw_signal_data['y_train'].to_numpy()

    m = KnnDtw(n_neighbors=1, max_warping_window=10)
    m.fit(x_train, y_train)
    # X test data -> raw_signal_data['total_acc_z_test']
    label, proba = m.predict(x_test[::10])

    # Y test data
    classification_report(label, y_test,
                          target_names=[l for l in labels.values()])

    conf_mat = confusion_matrix(label, y_test)

    fig = plt.figure(figsize=(6, 6))
    width = np.shape(conf_mat)[1]
    height = np.shape(conf_mat)[0]

    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c > 0:
                plt.text(j - .2, i + .1, c, fontsize=16)

    cb = fig.colorbar(res)
    plt.title('Confusion Matrix')
    _ = plt.xticks(range(6), [l for l in labels.values()], rotation=90)
    _ = plt.yticks(range(6), [l for l in labels.values()])

    return m


# ---------------------------
# main
# ---------------------------
if __name__ == '__main__':
    pathlib.Path('graphs/fft').mkdir(parents=True, exist_ok=True)
    pathlib.Path('data/converted').mkdir(parents=True, exist_ok=True)

    convert_to_pickles = True
    number_of_houses = 6

    if convert_to_pickles:
        house_files = []
        for house_number in range(1, number_of_houses):
            pathlib.Path('data/converted/house_' + str(house_number)).mkdir(parents=True, exist_ok=True)
            save_house_files('house_' + str(house_number))

        for house_number in range(1, number_of_houses):
            original_signal = build_original_signal('house_' + str(house_number))
            original_signal.to_pickle('data/converted/house_' + str(house_number) + '/original_signal.pkl')

    k_nearest_result = []
    fft_result = []
    for house_number in range(1, number_of_houses):
        original_signal = panda.read_pickle('data/converted/house_' + str(house_number) + '/original_signal.pkl')
        k_nearest_result[house_number] = k_nearest_heighbour(original_signal)
        fft_result[house_number] = run_fft(original_signal, 10, 0.5)

