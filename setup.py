"""A setuptools based setup module.
for NeuroLit project for UWSEDS A'17
"""


from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='NeuroLit',

    version='1.1.0',

    description='NeuroLit Reading Data Analysis',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/UWSEDS-aut17/uwseds-group-neurolit',

    # Author details
    author='Clarke, Donnelly, & Kethireddy',

    # Choose your license
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='reading data_analysis survey_data',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),


    # List run-time dependencies
    install_requires=['fancyimpute', 'missingno'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'test': ['coverage'],
    },

)
