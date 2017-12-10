from neurolit.base.dataset import *
from shutil import rmtree
import neurolit as nlit
import pandas as pd
import numpy as np
import unittest
import os

class TestBase(unittest.TestCase):

    def test_data_folder(self):
        data_path = os.path.join(nlit.__path__[0],'data')
        reading_data_path = os.path.join(data_path,'readingfile.csv')
        survey_data_path = os.path.join(data_path,'surveyfile.csv')
        data=nlit.Dataset(token_file = 'token.txt')
        self.assertEquals(data.frame.shape,(280,39))
        self.assertTrue(os.path.exists(reading_data_path))
        self.assertTrue(os.path.exists(survey_data_path))

        data=nlit.Dataset(data_folder = '', token_file = 'token.txt')
        self.assertEquals(data.frame.shape,(280,39))
        self.assertTrue(os.path.exists(reading_data_path))
        self.assertTrue(os.path.exists(survey_data_path))

        # data_path = os.path.expanduser('~/Desktop/data')
        # reading_data_path = os.path.join(data_path,'readingfile.csv')
        # survey_data_path = os.path.join(data_path,'surveyfile.csv')
        # data=nlit.Dataset(data_folder = '~/Desktop/data',
        #                   token_file = 'token.txt')
        # self.assertEquals(data.frame.shape,(273,39))
        # self.assertTrue(os.path.exists(reading_data_path))
        # self.assertTrue(os.path.exists(survey_data_path))
        # rmtree(data_path)

    def test_variable_selection(self):
        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_metalabels='WJ',
                          metalabel_files='readingdata_metalabels.csv',
                          token_file = 'token.txt')
        self.assertEqual(list(data.frame.columns),data.metalabel_dict['WJ'])

        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_metalabels=['WJ','CTOPP'],
                          selected_features = ['Music Training','Homeschooled'],
                          metalabel_files=['readingdata_metalabels.csv',
                                           'surveydata_metalabels.csv'],
                          token_file = 'token.txt')
        self.assertEqual(list(data.frame.columns),
                         data.metalabel_dict['WJ'] +
                         data.metalabel_dict['CTOPP'] +
                         ['Music Training', 'Homeschooled'])

        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_features = 'Music Training',
                          token_file = 'token.txt')
        self.assertEqual(list(data.frame.columns),['Music Training'])


    def test_parse_metalabel_files(self):
        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_features = 'Music Training',
                          token_file = 'token.txt')
        data.parse_metalabel_files(os.path.join(nlit.__path__[0],
                                   'data', 'test_metalabels.csv'))
        self.assertEqual(set(data.metalabel_dict.keys()),
                         set(['Metalabel','HAPPY','SAD']))

    def test_add_drop_metalabels(self):
        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_features = 'Music Training',
                          token_file = 'token.txt')
        data.parse_metalabel_files(os.path.join(nlit.__path__[0],
                                   'data', 'readingdata_metalabels.csv'))
        data.add_metalabels('WJ')
        data.add_metalabels(['TOWRE','CTOPP'])
        self.assertEqual(sorted(list(data.frame.columns)),
                         sorted(['Music Training'] +
                         data.metalabel_dict['WJ'] +
                         data.metalabel_dict['TOWRE'] +
                         data.metalabel_dict['CTOPP']))

        data.drop_metalabels('WJ')
        self.assertEqual(sorted(list(data.frame.columns)),
                         sorted(['Music Training'] +
                         data.metalabel_dict['TOWRE'] +
                         data.metalabel_dict['CTOPP']))

        data.drop_metalabels(['TOWRE','CTOPP'])
        self.assertEqual(list(data.frame.columns),['Music Training'])


    def test_add_drop_features(self):
        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_features = 'Music Training',
                          token_file = 'token.txt')
        data.add_features('Homeschooled')
        data.add_features(['Dyslexia Diagnosis','WJ Oral Reading'])
        self.assertEqual(sorted(list(data.frame.columns)),
                         sorted(['Music Training'] +
                         ['Homeschooled'] + ['Dyslexia Diagnosis'] +
                         ['WJ Oral Reading']))

        data.drop_features('WJ Oral Reading')
        self.assertEqual(sorted(list(data.frame.columns)),
                         sorted(['Music Training'] +
                         ['Homeschooled'] + ['Dyslexia Diagnosis']))

        data.drop_features(['Homeschooled','Music Training'])
        self.assertEqual(list(data.frame.columns),['Dyslexia Diagnosis'])


    def test_impute_missing(self):
        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_metalabels='WJ',
                          metalabel_files='readingdata_metalabels.csv',
                          outcome_variable = 'Dyslexia Diagnosis',
                          missingness_threshold = 0.4,
                          max_missing_count = 1,
                          token_file = 'token.txt')
        self.assertTrue(data.frame.isnull().values.any())
        data = impute_missing(data)
        self.assertFalse(data.frame.isnull().values.any())


    def test_drop_missing(self):
        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_metalabels='WJ',
                          metalabel_files='readingdata_metalabels.csv',
                          outcome_variable = 'Dyslexia Diagnosis',
                          token_file = 'token.txt')

        missingness_threshold = 0.4
        percent_missing = data.frame.isnull().sum()/data.frame.shape[0]
        num_valid_columns = sum(1 for x in percent_missing if x < 0.4)
        data.drop_missing_cols(missingness_threshold)
        self.assertEqual(num_valid_columns,data.frame.shape[1])


    def test_normalize_data(self):
        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_metalabels='WJ',
                          metalabel_files='readingdata_metalabels.csv',
                          outcome_variable = 'Dyslexia Diagnosis',
                          missingness_threshold = 0.4,
                          max_missing_count = 1,
                          token_file = 'token.txt')

        data = impute_missing(data)
        data = normalize_data(data)
        self.assertTrue(np.isclose(data.frame.mean(),
                        np.zeros((1,data.frame.shape[1]))).all())
        print(data.frame.var())
        self.assertTrue(np.isclose(data.frame.var(),
                        np.ones((1,data.frame.shape[1])), atol=1e-02).all())
