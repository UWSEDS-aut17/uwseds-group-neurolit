## Component Design

| Component | Justification |
| :--------------: | :-------------: |
| FetchData | Necessary for data curation in all use cases |
| ExtractVar | Necessary for data curation, variable selection for all use cases |
| VisualizeIt | Necessary for visualizations in each use case to illustrate relationships |
| ValidateIt | Implements training and tests sets to bring power to results in all use cases |


Name, FetchData
- pulls datasets from repository and converts to python dataFrame for use in analysis
- Inputs: nonemit
- Outputs: two python dataFrames
- PseudoCode:
	- url1 = x, url2 = y
	- pull data from urls
	- convert data to python dataFrames

Name, ExtractVar
- asks user for variables of interest from each dataset to perform analysis
- Inputs: string array (survey variable names), string array (quant variable name)
- Outputs: dataFrame, condensed
- PseudoCode:
	- ask user for survey variables
	- ask user for quant variables
	- condense 2 dataFrame objects to one condensed dataFrame with variables of interest for all those with complete data

Name, VisualizeIt
- asks user for desired plot depending on data used, and output is a visualization
- Inputs: string
- Outputs: figure (format TBD)
- PseudoCode:
	- asks user for desired plotting style depending on the inclusion of neuroimaging data, qualitative data, or ML data.

Name, ValidateIt
- performs data validation by dividing data into training and test data, testing via cross-validation
- Inputs: none
- Outputs: final model fit
- PseudoCode
	- divides data into test data and training data
	- performs cross-validation (method TBD)
	- output best model fit
