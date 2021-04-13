"""
This is eye_tracking_data.py contains Class EyeTrackingData and EyeTrackingBatchData
and the main() execution code.
| AUTHOR |    UPDATE    |   EMAIL                                 |
| LiuQM  |  2020/10/06  | contact:liuqm19@mails.tsinghua.edu.cn   |

TODO:

BUG:

Tricky:

"""
#%%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from scipy import signal, ndimage

import utils.utils as utils
import preprocess.new_columns as nc


class EyeTrackingData:

    def __init__(self, experiment_conf, name, path):
        self.white_event = experiment_conf['white_event']
        self.black_event = experiment_conf['black_event']

        self.block_time = experiment_conf['block_time']
        self.trial_time = experiment_conf['trial_time']
        self.baseline_time = experiment_conf['baseline_time']
        self.trial_num = experiment_conf['trial_num']
        self.crop_time = experiment_conf['crop_time']
        self.fig_path = experiment_conf['root_dir'] + experiment_conf['fig_path']
        
        self.name = name
        self.path = path

        self.data, self.sample_rate = self.__get_raw_data()

        self.data = self.__time_reconstruction()

        # check the validility of data both on slides-level and subject-level
        self.slides_valid = self.__get_slides_valid()
        self.subject_valid = 1 # default value is valid for each subject

        self.data = self.__preprocess()

        self.baseline_pupil = self.__get_baseline_pupil()

        self.data = self.__baseline_corrected()

        self.split_data_white, self.split_data_black = self.__split_data()

    def __baseline_corrected(self) -> pd.DataFrame:
        """
        Corrected pupil data by baseline pupil diameter.

        Args:

        Returns:
            data. Should be pd.DataFrame and corrected by basline pupil diameter
            
        Raises:

        """
        data = (self.data - self.baseline_pupil) / self.baseline_pupil

        return data

    def __get_baseline_pupil(self) -> float:
        """
        Get one subject's baseline pupil diameter by averaging pupil diameter in baseline_time.

        Args:

        Returns:
            baseline_pupil. Should be float>0.0
            3.5 means baseline_pupil = 3.5 mm
            
            Return Example:
            3.5 

        Raises:

        """
        baseline_start = 0
        baseline_end = self.baseline_time*self.sample_rate

        baseline_data = self.data[baseline_start:baseline_end]

        baseline_pupil = baseline_data.mean()

        return baseline_pupil

    def __get_raw_data(self) -> (list, int):
        """
        Get one subject's eye-tracking data.

        Args:

        Returns:
            raw_data: table-like eye-tracking data
            sample_rate:

            Return Example:


        Raises:
            Error: path does not exist.
        """
        if os.path.exists(self.path) is not True:
            raise Exception("path does not exist. Please check {}!".format(self.path))

        raw_data = []
        start_index = 0
        sample_rate = 120  # default sample rate is 120

        with open(self.path, 'r') as f:
            for line in f.readlines():
                line = line.split('\t')
                raw_data.append(line)

        for i in range(len(raw_data)):
            if raw_data[i][0] == '## Sample Rate:':
                sample_rate = int(raw_data[i][1])
            if raw_data[i][0][0] != '#':
                start_index = i
                break

        columns = raw_data[start_index]
        raw_data = raw_data[start_index + 1:]
        raw_data = pd.DataFrame(raw_data, columns=columns)
        return raw_data, sample_rate

    def __time_reconstruction(self):
        """
        due to the particularity of eye-tracking data,
        should reconstruct its time series to normal situation.

        Args:
            self:

        Returns:
            data: time_reconstructed data

        Raises:
            Error: Metric missed!!!
        """
        if 'L Mapped Diameter [mm]' not in self.data.columns.tolist():
            raise Exception(
                "Metric missed!!! 'L Mapped Diameter [mm]'! from subject {}".format(self.name))
        elif 'R Mapped Diameter [mm]' not in self.data.columns.tolist():
            raise Exception(
                "Metric missed!!! 'R Mapped Diameter [mm]'! from subject {}".format(self.name))

        columns = ['Time', 'Type', 'L Mapped Diameter [mm]', 'R Mapped Diameter [mm]']

        msg_columns_name = self.data.columns[3]
        if msg_columns_name not in columns:
            columns.append(msg_columns_name)

        data = self.data[columns]

        # 1. Extract message from data.
        msg_index = list(data[data['Type'] == 'MSG'].index)

        msg = data.loc[msg_index, msg_columns_name]
        msg_start_index = msg[msg.str.contains('blank')].index[-1]
        msg_start_time = int(data.loc[msg_start_index, 'Time'])

        # 2. Drop useless head and message in data.
        data = data[msg_start_index:]
        msg_index = [i for i in msg_index if i >= msg_start_index]
        data.drop(msg_index, inplace=True)

        # 3. Time reconstruction and drop duplicated. (time is absolute time in microseconds）
        data['SamplesNumber'] = data['Time'].apply(
            lambda x: np.ceil((int(x) - msg_start_time) * self.sample_rate / 1e6).astype(int))
        data['Millisecond'] = data['Time'].apply(
            lambda x: (int(x) - msg_start_time) / 1e3)
        data.drop_duplicates(subset={'SamplesNumber'}, keep='first', inplace=True)

        # 4. Do index reconstruction to fill missed point.
        data = data[data.SamplesNumber <= self.sample_rate * self.block_time]  # drop tail
        data.index = data.SamplesNumber
        data = data.reindex(index=range(1, self.sample_rate * self.block_time + 1), fill_value=0)
        data.SamplesNumber = data.index

        # 5. return data
        data = data[['L Mapped Diameter [mm]']].astype(float)
        data.replace(0, np.nan, inplace=True)
        data = pd.Series(data['L Mapped Diameter [mm]'])
        return data

    def __preprocess(self, filter='lp'):
        """
        Do interpolate and filtering.

        Args:
            self

        Returns:
            preprocessed data

        Raises:

        """
        data = self.data.interpolate(method='linear', limit_direction='both', axis=0)
        data = pd.Series(signal.savgol_filter(data, window_length=5, polyorder=2, axis=0),
                         name=data.name)
        return data

    def plot_all_trials(self):
        """
        Plot pupil diameter changed with time.

        Args:
            self

        Return:
        
        Raises:
            Path does not exist. Please check it !!!
        """
        if not os.path.exists(self.fig_path):
            raise Exception("Path does not exist. Please check {} !!!".format(self.fig_path))

        fig_path_prefix = self.fig_path + self.name

        pupil = self.data.tolist()
        time = self.data.index / self.sample_rate
        time = time.tolist()
        fig, ax = plt.subplots(figsize=(25.00, 15.45))

        ax.plot(time, pupil, 'k')

        slide_interval = int(self.trial_time / 2)
        x, y = 0, 0
        for i in self.event:
            start = i * self.sample_rate
            stop = (i + slide_interval) * self.sample_rate
            y = pupil[start: stop]
            x = time[start: stop]
            ax.plot(x, y, 'g')

        # Create plots with pre-defined labels.
        ax.plot(x, y, 'g', label='Stimulus white-slide on')
        legend = ax.legend(loc='upper right', shadow=True, fontsize='xx-large')

        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('w')

        plt.title(self.name, fontsize=28)
        plt.xlabel('time [s]', fontsize=28)
        plt.ylabel('Pupil Diameter [mm]', fontsize=28)

        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        plt.savefig(fig_path_prefix + '-block.png')
        plt.close()

        num = 0
        fig, _ = plt.subplots(figsize=(25.00, 15.45))
        for num, i in enumerate(self.event):
            start = i * self.sample_rate
            stop = (i + slide_interval) * self.sample_rate
            y = pupil[start: stop]
            x = time[start: stop]

            ax = plt.subplot(3, 2, num + 1)
            ax.plot(x, y, '-g', label='Stimulus white-slide on')

            plt.title(self.name, fontsize=14)
            plt.xlabel('time [s]', fontsize=14)
            plt.ylabel('Pupil Diameter [mm]', fontsize=14)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            legend = ax.legend(loc='upper center', shadow=True, fontsize='medium')
            legend.get_frame().set_facecolor('w')

        plt.savefig(fig_path_prefix + '-' + str(num + 1) + '_trials.png')
        plt.close()

        for num, i in enumerate(self.event):
            start = i * self.sample_rate
            stop = (i + slide_interval) * self.sample_rate
            y = pupil[start: stop]
            x = time[start: stop]

            plt.title(self.name)
            plt.xlabel('time [s]')
            plt.ylabel('Pupil Diameter [mm]')

            plt.plot(x, y, '-g', label='Stimulus white-slide on')

            legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            legend.get_frame().set_facecolor('w')

            plt.savefig(fig_path_prefix + '-' + 'trial' + str(num + 1) + '.png')
            plt.close()

    def plot_valid_trails(self):
        """
        Plot pupil diameter changed with time for valid trials.

        Args:
            self

        Return:
        
        Raises:
            Path does not exist. Please check it !!!
        """
        if not os.path.exists(self.fig_path):
            raise Exception("Path does not exist. Please check {} !!!".format(self.fig_path))

        fig_path_prefix = self.fig_path + self.name

        xlabel = "Time (in sec)"
        ylabel = chr(916)+ " in Pupil Diameter \n from baseline \n (in %)"

        title_font = {
            'family' : 'Times New Roman',
            'weight' : 'bold',
            'size'   : 30,
        }
        font = {
            'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 24,
        }

        # plot white slides
        whilte_trial_num = self.split_data_white.columns.size
        
        self.split_data_white.plot(figsize=(19.20, 10.80))

        plt.xlabel(xlabel, fontdict=font)
        plt.xticks(family='Times New Roman', weight='normal', size=24)
        plt.ylabel(ylabel, fontdict=font)
        plt.yticks(family='Times New Roman', weight='normal', size=24)

        plt.title("Valid white slides", fontdict=title_font)
        plt.legend(prop=font)

        plt.savefig(fig_path_prefix + '-white-' + str(whilte_trial_num) + 'trials' + '.png')
        plt.close()

        # plot black slides
        black_trial_num = self.split_data_black.columns.size

        self.split_data_black.plot(figsize=(19.20, 10.80))
        plt.xlabel(xlabel, fontdict=font)
        plt.xticks(family='Times New Roman', weight='normal', size=24)
        plt.ylabel(ylabel, fontdict=font)
        plt.yticks(family='Times New Roman', weight='normal', size=24)
        plt.title("Valid black slides", fontdict=title_font)
        plt.legend(prop=font)
        plt.savefig(fig_path_prefix + '-black-' + str(black_trial_num) + 'trials' + '.png')
        plt.close()

    def plot_valid_trails_avg(self):
        """
        Plot average pupil diameter changed with time for valid trials.

        Args:
            self

        Return:
        
        Raises:
            Path does not exist. Please check it !!!
        """
        if not os.path.exists(self.fig_path):
            raise Exception("Path does not exist. Please check {} !!!".format(self.fig_path))

        fig_path_prefix = self.fig_path + self.name

        xlabel = "Time (in sec)"
        ylabel = chr(916)+ " in Pupil Diameter \n from baseline \n (in %)"

        title_font = {
            'family' : 'Times New Roman',
            'weight' : 'bold',
            'size'   : 30,
        }
        font = {
            'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 24,
        }

        # plot white slides
        whilte_trial_num = self.split_data_white.columns.size
        white_data = self.split_data_white.mean(axis=1)

        white_data.plot(figsize=(19.20, 10.80))

        plt.xlabel(xlabel, fontdict=font)
        plt.xticks(family='Times New Roman', weight='normal', size=24)
        plt.ylabel(ylabel, fontdict=font)
        plt.yticks(family='Times New Roman', weight='normal', size=24)

        plt.title("Valid white slides (Average)", fontdict=title_font)

        plt.savefig(fig_path_prefix + '-white-' + str(whilte_trial_num) + 'trials' + "_avg" + '.png')
        plt.close()

        # plot black slides
        black_trial_num = self.split_data_black.columns.size
        black_data = self.split_data_black.mean(axis=1)

        black_data.plot(figsize=(19.20, 10.80))

        plt.xlabel(xlabel, fontdict=font)
        plt.xticks(family='Times New Roman', weight='normal', size=24)
        plt.ylabel(ylabel, fontdict=font)
        plt.yticks(family='Times New Roman', weight='normal', size=24)

        plt.title("Valid white slides (Average)", fontdict=title_font)

        plt.savefig(fig_path_prefix + '-black-' + str(black_trial_num) + 'trials' + "_avg" + '.png')
        plt.close()

    def __get_slides_valid(self):
        slides_valid = []

        for t in range(self.baseline_time, self.block_time, int(self.trial_time / 2)):
            start = t * self.sample_rate
            stop = (t + self.crop_time) * self.sample_rate
            pupil = self.data[start:stop]
            rate_null = pupil.isnull().sum() / pupil.shape[0]

            if rate_null > 0.25:
                # 数据缺失率 > 25%， 则视为无效数据
                slides_valid.append(0)
            else:
                slides_valid.append(1)

        return slides_valid

    def __split_data(self):
        """
        Split data into several white and black slides.

        Args:

        Return:
            split_data_white. Should be pd.DataFrame
            split_data_black. Should be pd.DataFrame            

        Raises:

        """
        split_data_white = pd.DataFrame([])
        split_data_black = pd.DataFrame([])
        idx_white = 1
        idx_black = 1
        slide_interval = int(self.trial_time / 2)
        for t in range(self.baseline_time, self.block_time, slide_interval):
            start = t * self.sample_rate
            stop = (t + self.crop_time) * self.sample_rate

            trial_index = (t - self.baseline_time) // slide_interval

            if self.slides_valid[trial_index]:
                if t in self.white_event:
                    split_data_white['white slide on:  ' + str(t) + 's-' + str(t+slide_interval) + 's'] = self.data[start:stop].values
                    
                elif t in self.black_event:
                    split_data_black['black slide on:  ' + str(t) + 's-' + str(t+slide_interval) + 's'] = self.data[start:stop].values

        split_data_white.index = split_data_white.index / self.sample_rate
        split_data_black.index = split_data_black.index / self.sample_rate

        return split_data_white, split_data_black

    @staticmethod
    def __get_features(split_data, sample_rate):
        """
        Get features from crops of time-series data.

        Args:
            split_data: input data [DataFrame]
            sample_rate:

        Return:

        Raises:

        """
        max_diameter = split_data.max(axis=0)
        min_diameter = split_data.min(axis=0)

        max_diameter_t = split_data.idxmax(axis=0)
        min_diameter_t = split_data.idxmin(axis=0)

        delta_diameter = abs(max_diameter - min_diameter)

        delta_t = abs(min_diameter_t - max_diameter_t)
        average_velocity = delta_diameter / delta_t

        features = [max_diameter, min_diameter, max_diameter_t, min_diameter_t,
                    delta_diameter, delta_t, average_velocity]

        features_name = ['max_diameter', 'min_diameter', 'max_diameter_t', 'min_diameter_t',
                         'delta_diameter', 'delta_t', 'average_velocity']

        statistics_name = ['num-valid-slides', 'mean_', 'std_', 'min_', 'middle_', 'max_']

        features = pd.DataFrame(features, index=features_name)

        features = features.T
        features_statistics = features.describe(percentiles=[.5])
        features_statistics.index = statistics_name
        
        return_features = EyeTrackingData.__flat(features_statistics)
        return return_features

    def extract_feature(self):
        if self.split_data_white.empty or self.split_data_black.empty:
            return pd.Series([], dtype=float)

        # get white slide and black slide features
        white_features = EyeTrackingData.__get_features(
            split_data=self.split_data_white, sample_rate=self.sample_rate)
        black_features = EyeTrackingData.__get_features(
            split_data=self.split_data_black, sample_rate=self.sample_rate)

        if white_features.empty or black_features.empty:
            return pd.Series([], dtype=float)

        def add_idx_title(index, title='w'):
            new_index = []
            for i in index.values:
                new_index.append(title + '_' + i)
            new_index = pd.Series(new_index)
            return new_index

        white_features.index = add_idx_title(white_features.index, 'w')
        black_features.index = add_idx_title(black_features.index, 'b')
        return_features = white_features.append(black_features)

        # get plr_latency
        # latency = self.__get_plr_latency()

        # return_features = return_features.append(latency)

        # get baseline pupil
        return_features['baseline_diameter'] = self.baseline_pupil
        
        return return_features

    def __get_plr_latency(self):
        """
        Get latency of Pupil Light Reflection Experiment from first 1s of crops of time-series data.

        Args:
            self

        Return:
            latency: unit s

        Raises:

        """
        position = self.split_data_white.loc[:int(self.sample_rate), :]
        velocity = position.diff()
        filtered_velocity = EyeTrackingData.__gussian_filter(self.sample_rate, pf=50, x=velocity)
        acceleration = velocity.diff()
        filtered_acc = filtered_velocity.diff()

        latency = filtered_acc.idxmax() / self.sample_rate
        latency.name = 'latency'
        latency_statistics = latency.describe(percentiles=[.5])
        latency_statistics.index = ['num-valid-slides', 'mean_', 'std_', 'min_', 'middle_', 'max_']
        latency_statistics = latency_statistics.drop(['num-valid-slides'])
        new_idx = []
        for idx in latency_statistics.index.values:
            new_idx.append(idx + 'plr_latency')
        latency_statistics.index = new_idx
        return latency_statistics

    @staticmethod
    def __gussian_filter(fs, pf, x):
        """
        Filtering data x by a gussian filter.

        Args:
            fs: sample rate of data x
            pf: parameter defined by user.
            x: input data [DataFrame]

        Return:
            filtered_x: filtered data.

        Raise:


        """
        w = fs * pf / 700
        p = np.round(3 * w)
        # c = np.round(p/2)
        sigma = w / 2
        truncate = ((p - 1) / 2 - 0.5) / sigma
        filtered_x = ndimage.gaussian_filter1d(input=x,
                                               sigma=sigma,
                                               axis=0,
                                               mode='nearest',
                                               truncate=truncate)
        filtered_x = pd.DataFrame(filtered_x, columns=x.columns)
        return filtered_x

    @staticmethod
    def __flat(data_frame) -> pd.Series:
        """
        Flat a DataFrame object into a Series object.
        And if num-valid_slides < 3, abandon this  data_frame

        Args:
            data_frame： an object of DataFrame class, contains several features.

        Return:
            flatted: an object of Series class, contains flatted features.

        Raises:
            Input Error:
                1.data_frame is not a pandas.DataFrame object.
                2.data_frame don't have the critical index: num-valid-slides.
        """
        if not isinstance(data_frame, pd.DataFrame):
            raise Exception(
                "Input Error: data_frame is not a pandas.DataFrame object. Please check!")
        if 'num-valid-slides' not in data_frame.index.values:
            raise Exception(
                "Input Error: data_frame don't have the critical index. Please check!")

        flatted = pd.Series([], dtype=float)
        for row in data_frame.index.values:
            if row == 'num-valid-slides':
                num_valid_slides = data_frame.loc[row, :].mean()
                if num_valid_slides < 3:
                    return flatted
                flatted[row] = num_valid_slides
                continue
            for col in data_frame.columns.values:
                flatted[row + col] = data_frame.loc[row, col]

        return flatted


class EyeTrackingBatchData:

    def __init__(self, experiment_conf):
        self.experiment_conf = experiment_conf
        self.batch_data = self.__get_batch_data()

    def __get_batch_data(self):
        batch_data = []

        path_dict = self.__get_filepath()

        for name, path in path_dict.items():
            batch_data.append(EyeTrackingData(self.experiment_conf, name, path))

        return batch_data

    def __get_filepath(self):
        """
        Args:

        returns:
            path_dict: {subject_num1 : data_path1, subject_num2 : data_path2, ...}

            Return Example:

        Raises:
            Error: Path does not exist.
        """
        data_path = self.experiment_conf['root_dir'] + self.experiment_conf['data_path']
        if os.path.exists(data_path) is not True:
            raise Exception("Path does not exist. Please check {}!".format(data_path))

        file = os.listdir(data_path)
        path_dict = {}

        for i in file:
            if i[-3:] != 'txt':
                continue
            subject = i.split('-')[0]
            path_dict[subject] = data_path + '/' + i

        return path_dict

    def plot_batch_data(self):
        """
        Usage:
            1.plot whole data, use data.plot_all_data().
            2.plot valid data, use data.plot_valid_data().
        """
        for data in self.batch_data:
            if data.subject_valid == 1:
                data.plot_valid_trails()
                data.plot_valid_trails_avg()

    def get_batch_features(self) -> pd.DataFrame:
        feature_list = []
        for data in self.batch_data:
            feature = data.extract_feature()
            if feature.empty:
                data.subject_valid = 0
                print("        Subject {}'s data is not valid. Drop it.".format(data.name))
                continue
            feature['name'] = data.name
            feature['label'] = EyeTrackingBatchData.__get_label(data.name)
            feature_list.append(feature)

        return_features = pd.DataFrame(feature_list)
        return return_features

    @staticmethod
    def __get_label(name) -> int:
        if name[0] != 's':
            char_label = name[0].lower()
            if char_label < 'j':
                return 1
            else:
                return 0
        else:
            char_label = name[1].lower()
            if char_label < 'j':
                return 1
            else:
                return 0


def get_all_features(exs_conf) -> pd.DataFrame:
    """
    Get features.

    Args:
        exs_conf: [Dictionary], experiments configuration.

    Return:
        all_features: [DataFrame], features are selected in eta.

    Raise:

    """
    all_features = pd.DataFrame([])

    for ex_name, ex_conf in exs_conf.items():
        print("For batch: {}".format(ex_conf['data_path']))
        print("    Read data...")
        eye_tracking_batch_data = EyeTrackingBatchData(ex_conf)


        print("    Get features...")
        feature_table = eye_tracking_batch_data.get_batch_features()

        # if you want to draw pictures of batch data use plot_batch_data()
        # eye_tracking_batch_data.plot_batch_data()

        print("    Done!!! \n")
        all_features = pd.concat([all_features, feature_table], axis=0)

    all_features['Count'] = all_features['valid count in white pictures'] + \
                                                all_features['valid count in black pictures']
                                                
    all_features = all_features.drop(columns=['valid count in white pictures', 'valid count in black pictures'])

    return all_features


def main():
    config_file = "D:/src/AutismDetection/AutismDetection/docs/plr_config.json"  # need changed by user

    exs_conf, output_conf = utils.get_config(config_file)

    all_features = get_all_features(exs_conf)

    all_features.to_csv(output_conf['all_features'], index=False)


if __name__ == '__main__':
    main()

# %%
