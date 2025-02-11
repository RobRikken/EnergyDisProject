# ---------------------------
# import
# ---------------------------

import pandas as panda
import numpy as np
from os import walk
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pathlib
import re
from multiprocessing import Process, cpu_count, Queue
from sup_learning import fully_connected_network
import glob
from keras.models import load_model
# environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from itertools import chain

#tabeas_path = '/Users/tabearoeber/Library/Mobile Documents/com~apple~CloudDocs/Uni/Utrecht/Semester3/Data Science/ED Project/EnergyDisProject/'
tabeas_path = ''


def load_data_file(path: str):
    return panda.read_csv(
        path,
        sep=' ',
        header=None
    )


def save_house_files(house: str) -> None:
    (_, __, file_names) = next(walk(tabeas_path + 'data/low_freq/' + house + '/'))
    file_names.remove('labels.dat')

    for file_name in file_names:
        signal_dataframe = load_data_file(tabeas_path + 'data/low_freq/' + house + '/' + file_name)
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
        labels_file = load_data_file(tabeas_path + 'data/low_freq/' + house_name + '/labels.dat')
        labels = panda.Series(labels_file[1])
        labels.index = labels_file[0]
        (_, _, file_names) = next(walk(tabeas_path + 'data/converted/' + house_name + '/'))
        for file_name in file_names:
            appliance_number = file_name.split("_")[1]
            appliance_number = int(re.sub("\.pkl$", '', appliance_number))
            appliance_name = labels[appliance_number] + '__' + str(appliance_number)
            files[house_name][appliance_name] = panda.read_pickle('data/converted/' + house_name + '/' + file_name)

    return files


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


def fit_distribution_to_label():
    plt.rcParams['figure.figsize'] = (16.0, 12.0)
    plt.style.use('ggplot')


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

            if 'refrigerator' in appliance:
                # X_train should be mains, Y_train is appliance
                appliance_series = house_files[house][appliance]
                selected_mains = panda.merge(
                    appliance_series,
                    combined_mains,
                    how='inner',
                    left_index=True,
                    right_index=True
                )
                # part_of_frame_end = round(len(appliance_series.index) * 0.7)
                # selected_mains = selected_mains.iloc[0:part_of_frame_end, :]
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


def process_data(df, dates, x_features, y_features, look_back=50):
    i = 0
    for date in dates:
        data = df.loc[date]
        len_data = data.shape[0]
        x = np.array([data[x_features].values[i:i + look_back]
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


def generate_dates(ddata_with_time_index: panda.DataFrame) -> List:
    dates = [str(time)[:10] for time in ddata_with_time_index.index.values]
    return sorted(list(set(dates)))


def read_merge_data(house):
    path = tabeas_path + 'data/low_freq/house_{}/'.format(house)
    file = path + 'channel_1.dat'
    df = panda.read_table(file, sep = ' ', names = ['unix_time', labels[house][1]], 
                                       dtype = {'unix_time': 'int64', labels[house][1]:'float64'}) 
    
    num_apps = len(glob.glob(path + 'channel*'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = panda.read_table(file, sep = ' ', names = ['unix_time', labels[house][i]], 
                                       dtype = {'unix_time': 'int64', labels[house][i]:'float64'})
        df = panda.merge(df, data, how = 'inner', on = 'unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time','timestamp'], axis=1, inplace=True)
    return df


def read_label():
    label = {}
    for i in range(1, 7):
        hi = tabeas_path + 'data/low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label


def process_multiple_houses_fcnn(houses_to_train_on: List[str]):
    # Now we combine three houses, and make on training set out of them.
    house_files = load_house_files()
    # House numbers are string because later they are used as a substring to check for.
    combined_mains = {}
    dates_combined_houses = {}
    d = []
    for house in house_files:
        # Get the house number for the string and check if it is in the houses to train.
        if house.split('_')[1] in houses_to_train_on:
            # Give both of the columns a different name, so we can select later.
            house_files[house]['mains__1'].name = 'mains_1'
            house_files[house]['mains__2'].name = 'mains_2'
            # The mains are here combined into a dataframe, so it can be joined in one go to the appliance.
            combined_mains[house] = panda.concat([house_files[house]['mains__1'], house_files[house]['mains__2']],
                                                 axis=1)
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

            dates_combined_houses = [str(time)[:10] for time in combined_mains[house].index.values]
            dates_combined_houses = sorted(list(set(dates_combined_houses)))
            d.append(dates_combined_houses)

    d = list(chain.from_iterable(d))
    d = np.unique(d)

    #dates = np.unique(combined_mains[house].index.date)
    first = True
    for date in d:
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


# def calculate mean square error
def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y))


# def calculate mean absolute error
def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y))


def plot_each_app(df, dates, predict, y_test, title, look_back = 0):
    num_date = len(dates)
    fig, axes = plt.subplots(num_date,1,figsize=(24, num_date*5) )
    plt.suptitle(title, fontsize = '25')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    for i in range(num_date):
        if i == 0: l = 0
        ind = df.loc[dates[i]].index[look_back:]
        axes.flat[i].plot(ind, y_test[l:l+len(ind)], color = 'blue', alpha = 0.6, label = 'True value')
        axes.flat[i].plot(ind, predict[l:l+len(ind)], color = 'red', alpha = 0.6, label = 'Predicted value')
        axes.flat[i].legend()
        l = len(ind)
        

def predictions(model, test_house, appliance, plot = True):
    
    # get test data from test house
    df1_test = df[test_house]
    X_test = df[test_house][['mains_2','mains_1']].values
    y_test = df[test_house][appliance].values

    y_pred = model.predict(X_test).reshape(-1)
    
    mse_tree = mse_loss(y_pred, y_test)
    mae_tree = mae_loss(y_pred, y_test)
    
    print('Mean square error on test set: ', mse_tree)
    print('Mean absolute error on the test set: ', mae_tree)
    
    if plot == True:
        dates_test = {}
        dates_test = [str(time)[:10] for time in df1_test.index.values]
        dates_test = sorted(list(set(dates_test)))
        
        plot_each_app(df1_test, dates_test, y_pred, y_test, 
                      title= 'Real and predict '+ appliance + ' of house' + str(test_house))
        
    return y_pred, mse_tree, mae_tree


# ---------------------------
# main
# ---------------------------
if __name__ == '__main__':

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    pathlib.Path(tabeas_path + 'graphs/fft').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tabeas_path + 'data/converted').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tabeas_path + 'models/fully_connected_network').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tabeas_path + 'models/lstm').mkdir(parents=True, exist_ok=True)

    # -1 to have on thread on the cpu free for other usage. Remove for max performance.
    number_of_processes = 1

    convert_to_pickles = False
    number_of_houses = 6

    if convert_to_pickles:
        house_files = []
        for house_number in range(1, number_of_houses + 1):
            pathlib.Path(tabeas_path + 'data/converted/house_' + str(house_number)).mkdir(parents=True, exist_ok=True)
            save_house_files('house_' + str(house_number))
    
    # use models to predict signal
    # get data     
    labels = read_label()
    for i in range(1, 7):
        print('House {}: '.format(i), labels[i], '\n')
    
    # merge data from channels into one df
    df = {}
    for i in range(1, 7): # for house 1 and 2
        df[i] = read_merge_data(i)
    # model 1: trained on house 1
    print("MODEL 1")
    # load model 
    fcnn_model1 = load_model(
        tabeas_path + "models/fully_connected_network/house_1/refrigerator__5_weights.198-8768.96.hdf5"
    )

    # train on other houses
    print("use model 1 on house 2")
    test_house = 2
    test_appliance_name = "refrigerator_9"

    test_house_predictions, test_house_mse, test_house_mae = predictions(
        fcnn_model1,
        test_house,
        test_appliance_name,
        plot=True
    )

    print("use model 1 on house 3")
    test_house = 3
    test_appliance_name = "refrigerator_7"
    
    test_house_predictions, test_house_mse, test_house_mae = predictions(
        fcnn_model1,
        test_house,
        test_appliance_name,
        plot=True
    )

    print("use model 1 on house 5")
    test_house = 5
    test_appliance_name = "refrigerator_18"
    
    test_house_predictions, test_house_mse, test_house_mae = predictions(
        fcnn_model1,
        test_house,
        test_appliance_name,
        plot=True
    )

    print("use model 1 on house 6")
    test_house = 6
    test_appliance_name = "refrigerator_8"
    
    test_house_predictions, test_house_mse, test_house_mae = predictions(
        fcnn_model1,
        test_house,
        test_appliance_name,
        plot=True
    )
    
    # model 2: trained on houses 1,2,3
    print("MODEL 2")
    fcnn_model2 = load_model(
        tabeas_path + 'models/fully_connected_network/house_1_2_3/refrigerator_weights.198-5262.53.hdf5'
    )
    

    print("use model 2 on house 5")
    test_house = 5
    test_appliance_name = "refrigerator_18"
    
    test_house_predictions, test_house_mse, test_house_mae = predictions(
        fcnn_model2,
        test_house,
        test_appliance_name,
        plot=True
    )

    print("use model 2 on house 6")
    test_house = 6
    test_appliance_name = "refrigerator_8"

    test_house_predictions, test_house_mse, test_house_mae = predictions(
        fcnn_model2,
        test_house,
        test_appliance_name,
        plot=True
    )
    
    
    # model 3: trained on houses 3,5,6
    print("MODEL 3")
    fcnn_model3 = load_model(
        tabeas_path + "models/fully_connected_network/house_3_5_6/refrigerator_weights.199-7313.56.hdf5"
    )
    
    print("use model 3 on house 1")
    test_house = 1
    test_appliance_name = "refrigerator_5"
    
    test_house_predictions, test_house_mse, test_house_mae = predictions(
        fcnn_model3,
        test_house,
        test_appliance_name,
        plot=True
    )
    
    print("use model 3 on house 2")
    test_house = 2
    test_appliance_name = "refrigerator_9"
    
    test_house_predictions, test_house_mse, test_house_mae = predictions(
        fcnn_model3,
        test_house,
        test_appliance_name,
        plot=True
    )
