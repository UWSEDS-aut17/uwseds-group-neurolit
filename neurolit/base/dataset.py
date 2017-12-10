from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from fancyimpute import KNN
from pathlib import Path
import missingno as msno
import neurolit as nlit
import seaborn as sns
import pandas as pd
import numpy as np
import requests
import sys
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

        selected_features (list/str): list of strings containing names
        of individual features from reading and survey data which
        are desired to be added to the dataset. Note: These are
        used as predictors

        selected_metalabels (list/str): list of strings containing names
        of metalabels associated with reading and survey data which
        are desired to be added to the dataset. Note: These are used
        as predictors

        outcome_variable (str): name of variable desired to be predicted
        as an outcome for building supervised learning models

        missingness_threshold (float): Real number between 0 and 1 which
        specifies the maximum perecent of missing values columns of data
        can have to be included in the Dataset

        max_missing_count (int): specifies the maximum number of missing
        data values in a row of the data. Any row with more missing values
        than max_missing_count will be filtered out

        metalabel_files (list): list of strings containing names of
        metalabel files in data_folder

        token_file (str): name of .txt file which contains API token
    """

    #Constructor for Dataset objects
    def __init__(self, data_folder = None,
                 output_folder = 'output',
                 selected_features = None,
                 selected_metalabels = None,
                 outcome_variable = None,
                 missingness_threshold = None,
                 max_missing_count = None,
                 metalabel_files = None,
                 token_file = None):


        self.features_list = []

        if data_folder is not None:
            if type(data_folder) is not str:
                raise TypeError("Warning: data_folder should be a string")
            data_folder = os.path.expanduser(data_folder)

        if data_folder is None or not os.path.exists(data_folder):
            print("Warning: specified data_folder does not exist")
            data_folder = os.path.join(nlit.__path__[0],'data')
            print("Data will be added to neurosynth/data directory")

        if metalabel_files is not None:
            if isinstance(metalabel_files, str):
                metalabel_files = [metalabel_files]
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

        reading_data.rename(columns={'record_id':'ID',
             'redcap_event_name':'Event', 'wj_lwid_ss':'WJ Letter Word ID',
             'wj_wa_ss':'WJ Word Attack', 'wj_or_ss':'WJ Oral Reading',
             'wj_srf_ss':'WJ Sentence Reading Fluency',
             'wj_mff_ss':'WJ Math Facts Fluency',
             'wj_brs':'WJ Basic Reading Skills', 'wj_rf':'WJ Reading Fluency',
             'twre_swe_ss':'TOWRE Sight Word Efficiency',
             'twre_pde_ss':'TOWRE Phonemic Decoding Efficiency',
             'twre_index':'TOWRE Index', 'wasi_vocab_ts':'WASI Vocabulary',
             'wasi_mr_ts':'WASI Matrix Reasoning',
             'wasi_fs2':'WASI Full Scale 2', 'ctopp_elision_ss':'CTOPP Elision',
             'ctopp_bw_ss':'CTOPP Blending Words',
             'ctopp_pi_ss':'CTOPP Phoneme Isolation',
             'ctopp_md_ss':'CTOPP Memory for Digits',
             'ctopp_nr_ss':'CTOPP Nonword Repetition',
             'ctopp_rd_ss':'CTOPP Rapid Digit Naming',
             'ctopp_rl_ss':'CTOPP Rapid Letter Naming',
             'ctopp_pa':'CTOPP Phonological Awareness',
             'ctopp_pm':'CTOPP Phonological Memory',
             'ctopp_rapid':'CTOPP Rapid Naming'}, inplace=True)

        survey_data.rename(columns={'record_id':'ID',
            'redcap_event_name':'EVENT', 'dys_dx':'Dyslexia Diagnosis',
            'dys_treat':'Dyslexia Treatment',
            'reading_rate':'Perceived Reading Skill',
            'school_grade': 'Grade', 'school_yn':'Enrolled in School',
            'private_school':'Private School',
            'homeschooled':'Homeschooled', 'read_prob': 'Problems with Reading',
            'repeated_grade':'Repeated a Grade',
            'read_train':'Reading Training',
            'dys_ass':'Assessed for Dyslexia', 'videogames':'Plays Videogames',
            'music_ed':'Music Education',
            'music_training':'Music Training'}, inplace=True)

        self.all_data = reading_data.set_index('ID').\
        join(survey_data.set_index('ID'),
        lsuffix='_reading', rsuffix='_survey')

        if not selected_features and not selected_metalabels:
            self.frame = self.all_data
        else:
            self.filter_data()

        if missingness_threshold is not None:
            self.drop_missing_cols(missingness_threshold)

        if max_missing_count is not None:
            if type(max_missing_count) is not int:
                raise TypeError("Warning: max_missing_count should be an int")
            self.drop_missing_rows(max_missing_count)

        if outcome_variable is not None:
            if type(outcome_variable) is not str:
                raise TypeError("Warning: outcome_variable should be a string")
            self.class_label = self.all_data[outcome_variable]


    def parse_metalabel_files(self, metalabel_files):

        """ Parse user-defined metalabel files and construct a dictionary
        of the metalabels and their corresponding features

        Args:
            metalabel_files (list/str): list of strings or a single string
            containing paths to metalabel files
        """

        if isinstance(metalabel_files, str):
            metalabel_files = [metalabel_files]
        metalabel_frame = pd.concat([pd.read_csv(f, header = None)
                                     for f in metalabel_files],
                                     ignore_index=True)
        self.metalabel_dict = {k: g.iloc[:,0].tolist()
        for k,g in metalabel_frame.groupby(metalabel_frame.iloc[:,1])}


    def add_metalabels(self, selected_metalabels):

        """ Add features, corresponding to specified metalabels, to dataset

        Args:
            selected_metalabels (list/str): list of strings or a single string
            containing the names of metalabels whose corresponding features
            will be added to the dataset
        """

        if isinstance(selected_metalabels, str):
            selected_metalabels = [selected_metalabels]
        for metalabel in selected_metalabels:
            self.add_features(self.metalabel_dict[metalabel])


    def add_features(self, selected_features):

        """ Add specified features to dataset

        Args:
            selected_features (list/str): list of strings or a single string
            containing the names of features which will be added to the dataset
        """

        if isinstance(selected_features, str):
            selected_features = [selected_features]
        for feature in selected_features:
            if feature not in self.features_list:
                self.features_list.append(feature)

        if hasattr(self, 'frame'):
            self.filter_data()


    def drop_metalabels(self, selected_metalabels):

        """ Drop features, corresponding to specified metalabels, from dataset

        Args:
            selected_metalabels (list/str): list of strings or a single string
            containing the names of metalabels whose corresponding features
            will be dropped from the dataset
        """

        if isinstance(selected_metalabels, str):
            selected_metalabels = [selected_metalabels]
        for metalabel in selected_metalabels:
            self.drop_features(self.metalabel_dict[metalabel])


    def drop_features(self, selected_features):

        """ Drop specified features from dataset

        Args:
            selected_features (list/str): list of strings or a single string
            containing the names of features which will be dropped from
            the dataset
        """

        if isinstance(selected_features, str):
            selected_features = [selected_features]
        for feature in selected_features:
            if feature in self.features_list:
                self.features_list.remove(feature)

        if hasattr(self, 'frame'):
            self.filter_data()


    def print_dataset_features(self):

        """ Print features that are currently included in the dataset
            (i.e. self.frame)
        """

        if not self.frame.columns:
            print('No features were selected.')
        else:
            print(self.frame.columns)


    def print_unused_features(self):

        """ Print all available features that are currently not included
            in the dataset (i.e. self.frame)
        """

        print(set(self.all_data.columns) - set(self.frame.columns))


    def print_metalabel_features(self, selected_metalabels):

        """ Print all features corresponding to the specified metalabels

        Args:
            selected_metalabels (list/str): list of strings or a single string
            containing the names of metalabels whose corresponding features
            will be printed
        """

        if isinstance(selected_metalabels, str):
            selected_metalabels = [selected_metalabels]
        if hasattr(self, 'metalabel_dict'):
            for metalabel in selected_metalabels:
                print(self.metalabel_dict[metalabel])
        else:
            print('Warning: Metalabel file(s) not provided.')


    def print_missingness(self):

        """ Print percentage of missing values in each column of the dataset
            (i.e. self.frame)
        """

        print(self.frame.isnull().sum()/self.frame.shape[0]*100)


    def filter_data(self):

        """ filter_data is a helper function which creates the dataset
            based on the set of metlabels and features provided by the user
        """

        self.frame = self.all_data[self.features_list]


    def visualize_missingness(self, output_directory,
                                    fig_name = 'missingness.png'):

        """ Visualize the distribution of missing data in the dataset using
            the missingno library

        Args:
            output_directory (str): specifies the path to the folder
            where the user wants to save the figure

            fig_name (str): specifies the name of the figure which
            will be stored in the specified output_directory
        """

        plt.figure()
        fig = msno.matrix(self.frame,inline=False)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        fig.savefig(os.path.join(output_directory, fig_name))


    def drop_missing_cols(self, missingness_threshold):

        """ Drop the columns with a smaller percentage of missing values
            than the specified missingness_threshold

        Args:
            missingness_threshold (float): a real number between 0 and 1
            which specifies the maximum tolerable percentage of missingness
            values in a column of data
        """

        missingness = self.frame.isnull().sum(axis=0)/self.frame.shape[0]
        drop_indexes = [i for i, m in enumerate(missingness)
        if m > missingness_threshold]
        self.frame.drop(self.frame.columns[drop_indexes],axis=1,inplace=True)


    def drop_missing_rows(self, max_missing_count):

        """ Drop the rows with a larger number of missing values
            than the specified max_missing_count

        Args:
            max_missing_count (int): an integer between 0 and the number
            of columns in the dataset which specifies the maximum tolerable
            number of missing values in a row of the dataset
        """

        drop_indexes = [i for i, m in enumerate(self.frame.isnull().sum(axis=1))
        if m > max_missing_count]
        self.frame.drop(self.frame.index[drop_indexes],axis=0,inplace=True)


def impute_missing(dataset_object):

    """ Uses a k-nearest neighbors algorithm with k = 5 to impute
        missing predictor values and the missing outcome variable
        if it is available
    """

    if hasattr(dataset_object,'class_label'):
        temp_data = dataset_object.frame.join(dataset_object.class_label)
        temp_data = KNN(k=5).complete(temp_data)
        dataset_object.class_label = pd.Series(data=np.round(temp_data[:,-1]),
                                             index = dataset_object.frame.index)
        dataset_object.frame = pd.DataFrame(data=temp_data[:,:-1],
                                        index = dataset_object.frame.index,
                                        columns = dataset_object.frame.columns)
    else:
        temp_data = KNN(k=5).complete(dataset_object.frame)
        dataset_object.frame = pd.DataFrame(data=temp_data,
                                        index = dataset_object.frame.index,
                                        columns = dataset_object.frame.columns)

    return dataset_object


def normalize_data(dataset_object):

    """ Normalizes the data by removing the mean and scaling each column
        to unit variance
    """

    temp_data = \
    StandardScaler().fit(dataset_object.frame).transform(dataset_object.frame)
    dataset_object.frame = pd.DataFrame(data=temp_data,
                                    index = dataset_object.frame.index,
                                    columns = dataset_object.frame.columns)
    return dataset_object
