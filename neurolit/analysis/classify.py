""" Classification tools """

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set()


class Classifier(object):

    """ Base class for curating data prior to analysis

    Insert more detailed description here.

    Args:
        dataset_object (Dataset): Dataset object created using the Dataset
        class in the dataset.py module. The dataset_object contains raw
        data, filtered data, and the outcome variable for training
        supervised learning models

        pca_data (DataFrame): optional argument specifying pca_data obtained
        from performing PCA, on the filtered data in the dataset_object, using
        the functions in the reduce.py module. If pca_data is specified, then
        the pca_data is used to train the supervised learning model instead
        of the filtered data in dataset_object

        percent_train (float): Real number between 0 and 1 which
        specifies the percentage of the data to use for training the
        machine learning model. The remaining data will be used to test the
        model accuracy

        model_type (str): name of desired classification model. Default is
        'random_forest' but 'logistic_regression' is also an option

    """

    #Constructor for Classifier objects
    def __init__(self, dataset_object = None,
                 pca_data = None,
                 percent_train = .8,
                 model_type = 'logistic_regression'):

        if pca_data is not None:
            self.features = pca_data
        else:
            self.features = dataset_object.frame

        self.labels = dataset_object.class_label

        self.X_train, self.X_test, self.y_train, self.y_test = \
        train_test_split(self.features, self.labels,
                         train_size=percent_train, random_state=42)


        if model_type is 'random_forest':
            self.classify_random_forest()

        if model_type is 'logistic_regression':
            self.classify_logistic_regression()


    def classify_random_forest(self):
        self.model = RandomForestClassifier(random_state=0, n_jobs=2)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)


    def classify_logistic_regression(self):
        self.model = LogisticRegression(random_state=0)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)


    def plot_confusion_matrix(self, output_directory,
                                    fig_name = 'confusionMatrix.png'):
        mat = confusion_matrix(self.y_test, self.y_pred)
        plt.figure()
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.savefig(os.path.join(output_directory, fig_name))
