""" Dimensionality reduction methods"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def perform_pca(dataset_object):
    pca = PCA().fit(dataset_object.frame)
    pca_data = pca.transform(dataset_object.frame)
    return pca, pca_data


#Plot of dataset variance captured by n principal components
def pca_variance_plot(pca, output_directory,
                           fig_name = 'PCAVariancePlot.png'):
    plt.figure()
    plt.plot((1- pca.explained_variance_ratio_), linewidth = 2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained variance')
    plt.title('Explained Variance vs. #Eigenmodes: PCA')
    plt.savefig(os.path.join(output_directory, fig_name))


#2D scatter plot of first and second PCA components
def pca_2D_scatter(pca_data, output_directory,
                            fig_name = 'PCA2DScatter.png'):
    plt.figure()
    plt.scatter(pca_data[:,0], pca_data[:,1])
    plt.title("First two PCA directions")
    plt.xlabel("1st eigenvector")
    plt.ylabel("2nd eigenvector")
    plt.savefig(os.path.join(output_directory, fig_name))


#3D scatter plot of first, second and third PCA components
def pca_3D_scatter(pca_data, output_directory,
                            fig_name = 'PCA3DScatter.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(pca_data[:,0], pca_data[:,1], pca_data[:,3])
    ax.set_xlabel('1st eigenvector')
    ax.set_ylabel('2nd eigenvector')
    ax.set_zlabel('3rd eigenvector')
    plt.savefig(os.path.join(output_directory, fig_name))
