from neurolit.base.dataset import *
from neurolit.analysis.reduce import *
from shutil import rmtree
import neurolit as nlit
import pandas as pd
import numpy as np
import unittest
import os

class TestBase(unittest.TestCase):

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
