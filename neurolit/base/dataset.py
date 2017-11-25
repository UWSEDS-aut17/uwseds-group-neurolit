import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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

        metalabel_files (list): list of strings containing names of
        metalabel files
    """

    #Constructor for Dataset objects
    def __init__(self, data_folder = 'data',
                 output_folder = 'output',
                 selected_features = None,
                 selected_metalabels = None,
                 metalabel_files = None):

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

        #NEED TO READ IN FROM API INSTEAD OF FILES
        reading_data = pd.read_csv(os.path.join(data_folder,
        'RDRPRepository_DATA_LABELS_2017-11-15_0943.csv'))
        survey_data = pd.read_csv(os.path.join(data_folder,
        'RDRPRepository_DATA_LABELS_2017-11-15_0944.csv'))

        self.all_data = reading_data.set_index('Record ID').\
        join(survey_data.set_index('Record ID'),
        lsuffix='_reading', rsuffix='_survey')

        self.filter_data()


    def parse_metalabel_files(self, metalabel_files):
        metalabel_frame = pd.concat([pd.read_csv(f) for f in metalabel_files])
        self.metalabel_dict = {k: g.iloc[:,0].tolist()
        for k,g in metalabel_frame.groupby(metalabel_frame.iloc[:,1])}


    def add_metalabels(self, selected_metalabels):
        for metalabel in selected_metalabels:
            self.add_features(self.metalabel_dict[metalabel])


    def add_features(self, selected_features):
        for feature in selected_features:
            if feature not in self.features_list:
                self.features_list.append(feature)

        if hasattr(self, 'frame'):
            self.filter_data()

    def drop_metalabels(self, selected_metalabels):
        for metalabel in selected_metalabels:
            self.drop_features(self.metalabel_dict[metalabel])


    def drop_features(self, selected_features):
        for feature in selected_features:
            if feature in self.features_list:
                self.features_list.remove(feature)

        if hasattr(self, 'frame'):
            self.filter_data()


    def print_dataset_features(self):
        if not self.features_list:
            print('No features were selected.')
        else:
            print(self.features_list)


    def print_unused_features(self):
        available_features = list(self.frame.columns)
        print(set(available_features) - set(self.features_list))


    def print_metalabel_features(self, selected_metalabels):
        if hasattr(self, 'metalabel_dict'):
            for metalabel in selected_metalabels:
                print(self.metalabel_dict[metalabel])
        else:
            print('Warning: Metalabel file(s) not provided.')

    def filter_data(self):
        self.frame = self.all_data[self.features_list]
