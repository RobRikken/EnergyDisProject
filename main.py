# ---------------------------
# import
# ---------------------------

import pandas as panda
import numpy as np
from os import walk
from os import environ
from typing import Dict
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.stats as stats
import pathlib
from k_nearest_neigbours import KnnDtw
from sklearn.metrics import classification_report, confusion_matrix
import re
from multiprocessing import Pool, Process, cpu_count, Queue
from sup_learning import fully_connected_network
import pickle
import time
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import glob
#environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf


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
    (_, house_names, _) = next(walk('data/converted/'))

    files = {}
    for house_name in house_names:
        files[house_name] = {}
        labels_file = load_data_file('data/low_freq/' + house_name + '/labels.dat')
        labels = panda.Series(labels_file[1])
        labels.index = labels_file[0]
        (_, _, file_names) = next(walk('data/converted/' + house_name + '/'))
        for file_name in file_names:
            appliance_number = file_name.split("_")[1]
            appliance_number = int(re.sub("\.pkl$", '', appliance_number))
            appliance_name = labels[appliance_number] + '__' + str(appliance_number)
            files[house_name][appliance_name] = panda.read_pickle('data/converted/' + house_name + '/' + file_name)

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
        sample_spacing: float = 0.01,
        subtract_mean: bool = False
) -> Dict:
    data_in_numpy_format = raw_signal_data.to_numpy()
    if subtract_mean:
        data_in_numpy_format = data_in_numpy_format - np.mean(data_in_numpy_format)

    yf = fft(data_in_numpy_format)
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


def k_nearest_neighbour(raw_signal_data: np.array, truth_values: np.array, house_name, appliance_name):

    model = KnnDtw(n_neighbors=5, max_warping_window=10000)
    model.fit(raw_signal_data, truth_values)

    pickle_file = open('models/k_nearest_neighbour/' + house_name + '_' + appliance_name + '.pkl', 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()

    return model


def report_fit_on_nearest_neighbours(fitted_neighbours: KnnDtw, x_test: panda.Series):
    # X test data -> raw_signal_data['total_acc_z_test']
    label, proba = fitted_neighbours.predict(x_test[::10])

    # Y test data
    classification_report(label, y_test,
                          target_names=[label for label in labels.values()]
                          )

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


def fit_distribution_to_label():
    plt.rcParams['figure.figsize'] = (16.0, 12.0)
    plt.style.use('ggplot')


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    size = 30000
    # x = scipy.arange(size)
    # y = scipy.int_(scipy.round_(stats.vonmises.rvs(5, size=size) * 47))
    # h = plt.hist(y, bins=range(48))

    dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        param = dist.fit(y)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
    #     plt.plot(pdf_fitted, label=dist_name)
    #     plt.xlim(0, 47)
    # plt.legend(loc='upper right')
    # plt.show()

    # Distributions to check
    DISTRIBUTIONS = [
        stats.alpha, stats.anglit, stats.arcsine, stats.beta, stats.betaprime, stats.bradford, stats.burr, stats.cauchy,
        stats.chi, stats.chi2,
        stats.cosine,
        stats.dgamma, stats.dweibull, stats.erlang, stats.expon, stats.exponnorm, stats.exponweib, stats.exponpow,
        stats.f, stats.fatiguelife,
        stats.fisk,
        stats.foldcauchy, stats.foldnorm, stats.frechet_r, stats.frechet_l, stats.genlogistic, stats.genpareto,
        stats.gennorm,
        stats.genexpon,
        stats.genextreme, stats.gausshyper, stats.gamma, stats.gengamma, stats.genhalflogistic, stats.gilbrat,
        stats.gompertz,
        stats.gumbel_r,
        stats.gumbel_l, stats.halfcauchy, stats.halflogistic, stats.halfnorm, stats.halfgennorm, stats.hypsecant,
        stats.invgamma,
        stats.invgauss,
        stats.invweibull, stats.johnsonsb, stats.johnsonsu, stats.ksone, stats.kstwobign, stats.laplace, stats.levy,
        stats.levy_l,
        stats.levy_stable,
        stats.logistic, stats.loggamma, stats.loglaplace, stats.lognorm, stats.lomax, stats.maxwell, stats.mielke,
        stats.nakagami, stats.ncx2,
        stats.ncf,
        stats.nct, stats.norm, stats.pareto, stats.pearson3, stats.powerlaw, stats.powerlognorm, stats.powernorm,
        stats.rdist,
        stats.reciprocal,
        stats.rayleigh, stats.rice, stats.recipinvgauss, stats.semicircular, stats.t, stats.triang, stats.truncexpon,
        stats.truncnorm,
        stats.tukeylambda,
        stats.uniform, stats.vonmises, stats.vonmises_line, stats.wald, stats.weibull_min, stats.weibull_max,
        stats.wrapcauchy
    ]


def execute_network_learning_queue(the_queue: Queue):
    while True:
        if the_queue.empty():
            break

        training_data = the_queue.get_nowait()
        fully_connected_network(
            training_data['X_train'],
            training_data['Y_train'],
            training_data['house_name'],
            training_data['appliance_name'],
        )

    return True


def execute_k_nearest_neighbour(the_queue: Queue):
    while True:
        if the_queue.empty():
            break

        training_data = the_queue.get_nowait()
        k_nearest_neighbour(
            training_data['X_train'],
            training_data['Y_train'],
            training_data['house_name'],
            training_data['appliance_name']
        )

    return True


def process_networked_learning():
    house_files = load_house_files()
    processes = []
    queue = Queue()
    # Fill the queue
    for house in house_files:
        house_files[house]['mains__1'].name = 'mains_1'
        house_files[house]['mains__2'].name = 'mains_2'
        combined_mains = panda.concat([house_files[house]['mains__1'], house_files[house]['mains__2']], axis=1)
        for appliance in house_files[house]:
            if 'mains' in appliance:
                continue

            if not ('refrigerator' in appliance):
                # X_train should be mains, Y_train is applian
                appliance_series = house_files[house][appliance]
                selected_mains = panda.merge(appliance_series, combined_mains, how='inner', left_index=True,
                                             right_index=True)
                queue.put({
                    'X_train': selected_mains[['mains_1', 'mains_2']].to_numpy(),
                    'Y_train': selected_mains['power'].to_numpy(),
                    'house_name': house,
                    'appliance_name': appliance
                })

    for process_number in range(number_of_processes):
        process = Process(target=execute_network_learning_queue, args=(queue,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def process_k_nearest_neighbours():

    house_files = load_house_files()
    processes = []
    queue = Queue()
    # Fill the queue
    for house in house_files:
        house_files[house]['mains__1'].name = 'mains_1'
        house_files[house]['mains__2'].name = 'mains_2'
        combined_mains = panda.concat([house_files[house]['mains__1'], house_files[house]['mains__2']], axis=1)
        for appliance in house_files[house]:
            if 'mains' in appliance:
                continue

            # X_train should be mains, Y_train is applian
            appliance_series = house_files[house][appliance]
            selected_signals = panda.merge(appliance_series, combined_mains, how='inner', left_index=True, right_index=True)
            single_main = selected_signals['mains_1'] + selected_signals['mains_2']
            queue.put({
                'X_train': single_main.to_numpy(),
                'Y_train': selected_signals['power'].to_numpy(),
                'house_name': house,
                'appliance_name': appliance
            })

    for process_number in range(number_of_processes):
        process = Process(target=execute_k_nearest_neighbour, args=(queue,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def build_lstm_model(layers):
    model = Sequential()
    for i in range(len(layers) - 2):
        model.add(LSTM(
            input_dim=layers[i],
            output_dim=layers[i + 1],
            # stateful=True,
            return_sequences=True if i < len(layers) - 3 else False))
        model.add(Dropout(0.3))

    model.add(Dense(layers[-1]))
    model.summary()

    return model


def fft_all_appliances():
    house_files = load_house_files()
    for house in house_files:
        for appliance in house_files[house]:
            if 'mains' in appliance:
                continue

            appliance_signal = house_files[house][appliance]
            fft_result = run_fft(
                house_files[house][appliance].to_frame(),
                number_of_samples=100000,
                sample_spacing=0.01,
                subtract_mean=True
            )
            fft_result['name'] = house + ' ' + appliance
            make_fft_plot(fft_result, 100000, house + '_' + appliance)


def process_data(df, dates, x_features, y_features, look_back = 50):
    i = 0
    for date in dates:
        data = df.loc[date]
        len_data = data.shape[0]
        x = np.array([data[x_features].values[i:i+look_back]
                      for i in range(len_data - look_back)]).reshape(-1, look_back, 2)
        y = data[y_features].values[look_back:, :]
        if i == 0:
            X = x
            Y = y
        else:
            X = np.append(X, x, axis=0)
            Y = np.append(Y, y, axis=0)
        i += 1

    return X, Y


def generate_dates():
    dates = {}
    for i in range(1, 3):
        dates[i] = [str(time)[:10] for time in df[i].index.values]
        dates[i] = sorted(list(set(dates[i])))
        print('House {0} data contain {1} days from {2} to {3}.'.format(i, len(dates[i]), dates[i][0], dates[i][-1]))
        print(dates[i], '\n')


def read_merge_data(house, labels):
    path = 'data/low_freq/house_{}/'.format(house)
    file = path + 'channel_1.dat'
    df = panda.read_table(file, sep=' ', names=['unix_time', labels[house][1]],
                       dtype={'unix_time': 'int64', labels[house][1]: 'float64'})

    num_apps = len(glob.glob(path + 'channel*'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = panda.read_table(file, sep=' ', names=['unix_time', labels[house][i]],
                             dtype={'unix_time': 'int64', labels[house][i]: 'float64'})
        df = panda.merge(df, data, how='inner', on='unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time', 'timestamp'], axis=1, inplace=True)
    return df


def read_label():
    label = {}
    for i in range(1, 7):
        hi = 'data/low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label


def run_with_gpu():
    labels = read_label()

    df = {}
    for i in range(1, 3):
        df[i] = read_merge_data(i, labels)

    dates = {}
    for i in range(1, 3):
        dates[i] = [str(time)[:10] for time in df[i].index.values]
        dates[i] = sorted(list(set(dates[i])))

    house_files = load_house_files()
    # Fill the queue
    for house in house_files:
        house_files[house]['mains__1'].name = 'mains_1'
        house_files[house]['mains__2'].name = 'mains_2'
        combined_mains = panda.concat([house_files[house]['mains__1'], house_files[house]['mains__2']], axis=1)
        for appliance in house_files[house]:
            if 'mains' in appliance:
                continue

            if 'refrigerator' in appliance:
                model = build_lstm_model([2, 64, 128, 256, 1])
                start = time.time()
                adam = Adam(lr=5e-5)
                model.compile(loss='mean_squared_error', optimizer=adam)
                checkpointer = ModelCheckpoint(
                    filepath="models/lstm/lstm_model_" + appliance + ".hdf5",
                    verbose=0,
                    save_best_only=True
                )

                # X_train should be mains, Y_train is applian
                appliance_series = house_files[house][appliance]
                selected_signals = panda.merge(appliance_series, combined_mains, how='inner', left_index=True,
                                               right_index=True)
                # single_main = selected_signals['mains_1'] + selected_signals['mains_2']

                X_train, y_train = process_data(df[1], dates[1][:17], ['mains_1', 'mains_2'], df[1].columns.values[2:])
                X_test, y_test = process_data(df[1], dates[1][17:], ['mains_1', 'mains_2'], df[1].columns.values[2:])

                hist_lstm = model.fit(
                    X_train,
                    y_train[:, 2],
                    batch_size=512,
                    verbose=1,
                    nb_epoch=200,
                    validation_split=0.3,
                    callbacks=[checkpointer])
                print('Finish trainning. Time: ', time.time() - start)


# ---------------------------
# main
# ---------------------------
if __name__ == '__main__':

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    pathlib.Path('graphs/fft').mkdir(parents=True, exist_ok=True)
    pathlib.Path('data/converted').mkdir(parents=True, exist_ok=True)
    pathlib.Path('models/fully_connected_network').mkdir(parents=True, exist_ok=True)
    pathlib.Path('models/lstm').mkdir(parents=True, exist_ok=True)

    # -1 to have on thread on the cpu free for other usage. Remove for max performance.
    number_of_processes = 1

    convert_to_pickles = False
    number_of_houses = 6

    if convert_to_pickles:
        house_files = []
        for house_number in range(1, number_of_houses + 1):
            pathlib.Path('data/converted/house_' + str(house_number)).mkdir(parents=True, exist_ok=True)
            save_house_files('house_' + str(house_number))

    # Now we combine three houses, and make on training set out of them.
    house_files = load_house_files()
    # House numbers are string because later they are used as a substring to check for.
    houses_to_train_on = ['3', '5', '6']
    combined_mains = {}
    for house in house_files:
        # Get the house number for the string and check if it is in the houses to train.
        if house.split('_')[1] in houses_to_train_on:
            # Give both of the columns a different name, so we can select later.
            house_files[house]['mains__1'].name = 'mains_1'
            house_files[house]['mains__2'].name = 'mains_2'
            # The mains are here combined into a dataframe, so it can be joined in one go to the appliance.
            combined_mains[house] = panda.concat([house_files[house]['mains__1'], house_files[house]['mains__2']], axis=1)
            for appliance in house_files[house]:
                # Do not train on the mains, they are the X_input.
                if 'mains' in appliance:
                    continue

                # Select and appliances here, or remove to train all appliances
                if 'refrigerator' in appliance:
                    # X_train should be mains, Y_train is appliance
                    appliance_series = house_files[house][appliance]
                    combined_mains[house] = panda.merge(
                        appliance_series,
                        combined_mains[house],
                        how='inner',
                        left_index=True,
                        right_index=True
                    )

    dates = np.unique(combined_mains[house].index.date)
    first = True
    for date in dates:
        for house in house_files:
            # Only run the houses that are selected to run, by checking if the number is in the list.
            if house.split('_')[1] in houses_to_train_on:
                if first:
                    houses_combined_signal = combined_mains[house].loc[str(date)]
                    first = False
                else:
                    houses_combined_signal = houses_combined_signal.append(combined_mains[house].loc[str(date)])

    x_train = houses_combined_signal[['mains_1', 'mains_2']]
    y_train = houses_combined_signal['power']

    combined_house_name = ''
    for number in houses_to_train_on:
        combined_house_name = combined_house_name + '_' + number

    fully_connected_network(x_train.to_numpy(), y_train.to_numpy(), 'house' + combined_house_name, 'refrigerator')

