from neurolit.base.dataset import *
from neurolit.analysis.reduce import *
from shutil import rmtree
import neurolit as nlit
import pandas as pd
import numpy as np
import unittest
import os

class TestAnalysis(unittest.TestCase):

    def test_perform_pca(self):
        data=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                          selected_metalabels='WJ',
                          metalabel_files='readingdata_metalabels.csv',
                          outcome_variable = 'Dyslexia Diagnosis',
                          missingness_threshold = 0.4,
                          max_missing_count = 1,
                          token_file = 'token.txt')
        data = impute_missing(data)
        data = normalize_data(data)
        self.assertEquals(data.frame.shape,(234,6))
        pca, pca_data = perform_pca(data)
        self.assertEquals(pca_data.shape,(234,6))
        pca_data = extract_pca_components(pca_data)
        self.assertEquals(pca_data.shape,(234,2))

    def test_build_classifier(self):
        data_full=nlit.Dataset(data_folder = os.path.join(nlit.__path__[0],'data'),
                               selected_metalabels='WJ',
                               metalabel_files='readingdata_metalabels.csv',
                               outcome_variable = 'Dyslexia Diagnosis',
                               missingness_threshold = 0.4,
                               max_missing_count = 1,
                               token_file = 'token.txt')
        data_full = impute_missing(data_full)
        data_full = normalize_data(data_full)
        pca, data_pca = perform_pca(data_full)
        data_pca = extract_pca_components(data_pca)

        classifier_a = nlit.Classifier(dataset_object = data_full)
        self.assertEquals(classifier_a.features.shape,(234,6))

        classifier_b = nlit.Classifier(dataset_object = data_full,
                                       pca_data = data_pca)
        self.assertEquals(classifier_b.features.shape,(234,2))
