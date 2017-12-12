## Component Design

| Component | Justification |
| :--------------: | :-------------: |
| Dataset | Necessary for data curation, variable selection for all use cases |
| Classify | Necessary for curating data prior to analysis |
| Reduce | Necessary for dimensionality reduction methods |
| dash_visualize.ipynb | Necessary for visualizations in each use case to illustrate relationships |
| predict.ipynb | Necessary for providing a UI for visualizing ML applications |
| tests_analysis | Necessary for testing the functionality of the analysis functions |
| tests_base | Necessary for testing the functionality of the data curation functions |

Name, Dataset
- pulls datasets from repository and converts to python dataFrame for use in analysis
- Inputs: data location string, output folder string, selected features str, selected metalabels str, outcome variable str, missingness threshold float, max missing count int, metalabel_files list
- Outputs: Dataset Classifier object
- PseudoCode:
	- set parameters for data extraction
	- pull data from database using API token
	- convert data to python dataFrame and sift based on user specifications

Name, Classify
- curates data prior to analysis
- Inputs: datase object, pca dataframe, percent training float, model type string
- Outputs: Dataset Classifier object, visualizations
- PseudoCode:
	- input specifications
	- run variety of classification techniques based on those specifications
	
Name, Reduce
-performs dimensionality reduction algorithms
-Inputs: Dataset Classifier object, pca specifications
-Outputs: Dataset classifier object, visualizations
-PseudoCode:
	- input specifications
	- run variety of dimensionality reduction techniques based on those specifications	-

Name, dash_visualize.ipynb
- run script to open dash app visualization tool
- Inputs: Dataset function specifications
- Outputs: dash app
- PseudoCode:
	- run dataset function
	- opens visualization tool

Name, predict.ipynb
- performs ML algorithms and displays visualizations
- Inputs: dataset function specifications
- Outputs: inline visualizations
- PseudoCode
	- input dataset specifications
	- run NeuroLit functions and display results
