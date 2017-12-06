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
        self.assertEquals(data.frame.shape,(273,39))
        self.assertTrue(os.path.exists(reading_data_path))
        self.assertTrue(os.path.exists(survey_data_path))

        data=nlit.Dataset(data_folder = '', token_file = 'token.txt')
        self.assertEquals(data.frame.shape,(273,39))
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
