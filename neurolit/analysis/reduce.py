""" Dimensionality reduction methods"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def perform_pca(dataset_object):

    """ Peform PCA and return both the "model" and the transformed data.
        Note that pca_data returns number of components equal to the number
        of features/columns of data

    Args:
        dataset_object (Dataset): Dataset object is created using the
        Dataset class in dataset.py
    """

    pca = PCA().fit(dataset_object.frame)
    pca_data = pca.transform(dataset_object.frame)
    pca_data = pd.DataFrame(data=pca_data,
                            index = dataset_object.frame.index)

    return pca, pca_data


def extract_pca_components(pca_data, n_components = 2):

    """ Extract specified number of principal components from pca_data
        created using the perform_pca function

    Args:
        pca_data (DataFrame): Pandas DataFrame object containing number
        of principal components equal to the number of data columns/features

        n_components (int): Number of pca components to be extracted from
        original pca_data
    """

    pca_data = pd.DataFrame(data=pca_data.iloc[:,:n_components],
                            index = pca_data.index)

    return pca_data


#Plot of dataset variance captured by n principal components
def pca_variance_plot(pca, output_directory,
                           fig_name = 'PCAVariancePlot.png'):

    """ Visualize the percent variance caputred vs number of principal
        components

    Args:
        pca (PCA): PCA object created using scikit-learn in the perform_pca
        function

        output_directory (str): output directory to place the figure in

        fig_name (str): name of figure to be placed in output_directory
    """

    plt.figure()
    plt.plot((1- pca.explained_variance_ratio_), linewidth = 2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained variance')
    plt.title('Explained Variance vs. #Eigenmodes: PCA')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(os.path.join(output_directory, fig_name))


#2D scatter plot of first and second PCA components
def pca_2D_scatter(pca_data, output_directory,
                            fig_name = 'PCA2DScatter.png'):

    """ Visualize the first two principal components

    Args:
        pca_data (DataFrame): Pandas DataFrame object containing number
        of principal components equal to the number of data columns/features

        output_directory (str): output directory to place the figure in

        fig_name (str): name of figure to be placed in output_directory
    """

    plt.figure()
    plt.scatter(pca_data.iloc[:,0], pca_data.iloc[:,1])
    plt.title("First two PCA directions")
    plt.xlabel("1st eigenvector")
    plt.ylabel("2nd eigenvector")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(os.path.join(output_directory, fig_name))


#3D scatter plot of first, second and third PCA components
def pca_3D_scatter(pca_data, output_directory,
                            fig_name = 'PCA3DScatter.png'):

    """ Visualize the first three principal components

    Args:
        pca_data (DataFrame): Pandas DataFrame object containing number
        of principal components equal to the number of data columns/features

        output_directory (str): output directory to place the figure in

        fig_name (str): name of figure to be placed in output_directory
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(pca_data.iloc[:,0], pca_data.iloc[:,1], pca_data.iloc[:,3])
    ax.set_xlabel('1st eigenvector')
    ax.set_ylabel('2nd eigenvector')
    ax.set_zlabel('3rd eigenvector')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(os.path.join(output_directory, fig_name))
