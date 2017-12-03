import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import pandas as pd
import numpy as np
import requests
import os

sns.set()

class Dataset(object):

    """ Base class for curating data prior to analysis

    Insert more detailed description here.

    Args:
        data_folder (str): name of folder which contains any raw data,
        and metalabel files

        output_folder (str): name of folder where any figures and data
        generated will be output

        selected_features (list): list of strings containing names
        of individual features from reading and survey data which
        are desired to be added to the dataset

        selected_metalabels (list): list of strings containing names
        of metalabels associated with reading and survey data which
        are desired to be added to the dataset

        missingness_threshold (float): Real number between 0 and 1 which
        specifies the maximum perecent of missing values columns of data
        can have to be included in the Dataset

        max_missing_count (int): specifies the maximum number of missing
        data values in a row of the data. Any row with more missing values
        than max_missing_count will be filtered out

        metalabel_files (list): list of strings containing names of
        metalabel files

        token_file (str): name of .txt file which contains API token
    """

    #Constructor for Dataset objects
    def __init__(self, data_folder = 'data',
                 output_folder = 'output',
                 selected_features = None,
                 selected_metalabels = None,
                 missingness_threshold = None,
                 max_missing_count = None,
                 metalabel_files = None,
                 token_file = None):

        self.features_list = []

        if metalabel_files is not None:
            metalabel_files = [os.path.join(data_folder, f)
            for f in metalabel_files]
            self.parse_metalabel_files(metalabel_files)

        if selected_metalabels is not None:
            if hasattr(self, 'metalabel_dict'):
                self.add_metalabels(selected_metalabels)
            else:
                print('Warning: Metalabel file(s) not found. The features',
                'corresponding to the specified metalabels were not added',
                'to the dataset.')

        if selected_features is not None:
            self.add_features(selected_features)

        # RedCap API data access
        # This code snippet will pull the data reports from RedCap as csv files
        # and convert to Python DataFrame objects

        if token_file is not None:
            token_path = os.path.join(data_folder, token_file)
            if os.path.exists(token_path):
                with open(token_path, 'r') as myfile:
                    token=myfile.read().replace('\n', '')
        else:
            token = input('What is the API token? ')

        reading_data = {
            'token': token,
            'content': 'report',
            'format': 'csv',
            'report_id': '20197',
            'rawOrLabel': 'raw',
            'rawOrLabelHeaders': 'raw',
            'exportCheckboxLabel': 'false',
            'returnFormat': 'csv'
        }
        survey_data = {
            'token': token,
            'content': 'report',
            'format': 'csv',
            'report_id': '20199',
            'rawOrLabel': 'raw',
            'rawOrLabelHeaders': 'raw',
            'exportCheckboxLabel': 'false',
            'returnFormat': 'csv'
        }

        redcap_path = 'https://redcap.iths.org/api/'
        r_reading = requests.post(redcap_path, data=reading_data)
        r_survey = requests.post(redcap_path, data=survey_data)

        reading_filename =os.path.join(data_folder,'readingfile.csv')
        with open(reading_filename, 'w') as reading_file:
            reading_file.write(r_reading.text)

        survey_filename = os.path.join(data_folder,'surveyfile.csv')
        with open(survey_filename, 'w') as survey_file:
            survey_file.write(r_survey.text)

        reading_data = pd.read_csv(reading_filename)
        survey_data = pd.read_csv(survey_filename)

        self.all_data = reading_data.set_index('record_id').\
        join(survey_data.set_index('record_id'),
        lsuffix='_reading', rsuffix='_survey')

        if not selected_features and not selected_metalabels:
            self.frame = self.all_data
        else:
            self.filter_data()

        if missingness_threshold is not None:
            self.drop_missing_cols(missingness_threshold)

        if max_missing_count is not None:
            self.drop_missing_rows(max_missing_count)


    def parse_metalabel_files(self, metalabel_files):
        metalabel_files = list(metalabel_files)
        metalabel_frame = pd.concat([pd.read_csv(f) for f in metalabel_files])
        self.metalabel_dict = {k: g.iloc[:,0].tolist()
        for k,g in metalabel_frame.groupby(metalabel_frame.iloc[:,1])}


    def add_metalabels(self, selected_metalabels):
        selected_metalabels = list(selected_metalabels)
        for metalabel in selected_metalabels:
            self.add_features(self.metalabel_dict[metalabel])


    def add_features(self, selected_features):
        selected_features = list(selected_features)
        for feature in selected_features:
            if feature not in self.features_list:
                self.features_list.append(feature)

        if hasattr(self, 'frame'):
            self.filter_data()

    def drop_metalabels(self, selected_metalabels):
        selected_metalabels = list(selected_metalabels)
        for metalabel in selected_metalabels:
            self.drop_features(self.metalabel_dict[metalabel])


    def drop_features(self, selected_features):
        selected_metalabels = list(selected_features)
        for feature in selected_features:
            if feature in self.features_list:
                self.features_list.remove(feature)

        if hasattr(self, 'frame'):
            self.filter_data()


    def print_dataset_features(self):
        if not self.frame.columns:
            print('No features were selected.')
        else:
            print(self.frame.columns)


    def print_unused_features(self):
        available_features = list(self.frame.columns)
        print(set(available_features) - set(self.features_list))


    def print_metalabel_features(self, selected_metalabels):
        selected_metalabels = list(selected_metalabels)
        if hasattr(self, 'metalabel_dict'):
            for metalabel in selected_metalabels:
                print(self.metalabel_dict[metalabel])
        else:
            print('Warning: Metalabel file(s) not provided.')


    def print_missingness(self):
        print(self.frame.isnull().sum()/self.frame.shape[0]*100)


    def filter_data(self):
        self.frame = self.all_data[self.features_list]


    def visualize_missingness(self, output_directory,
                                    fig_name = 'missingness.png'):
        plt.figure()
        fig = msno.matrix(self.frame,inline=False)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        fig.savefig(os.path.join(output_directory, fig_name))


    def drop_missing_cols(self, missingness_threshold):
        missingness = self.frame.isnull().sum(axis=0)/self.frame.shape[0]
        drop_indexes = [i for i, m in enumerate(missingness)
        if m > missingness_threshold]
        self.frame.drop(self.frame.columns[drop_indexes],axis=1,inplace=True)


    def drop_missing_rows(self, max_missing_count):
        drop_indexes = [i for i, m in enumerate(self.frame.isnull().sum(axis=1))
        if m > max_missing_count]
        self.frame.drop(self.frame.index[drop_indexes],axis=0,inplace=True)
