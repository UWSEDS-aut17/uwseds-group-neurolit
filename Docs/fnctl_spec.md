
## Functional Specification

# User Profile:

- Our users will be research scientists familiar with python, and specifically, Jupyter Notebook.
- Our users will be familiar with the various reading measures, aptitude tests, and brain imaging data (MEG).
- Our users will also be familiar with the various survey questions used to create the survey dataset.

# Elements of the problem statement:
    
1. Can we create a tool that will produce fast, easy, and presentation-worthy visualizations of relationships in our data?
2. Using our survey data, do parent perceptions of reading difficulty correlate with assessed reading skill?
3. Is our reading assessment able to predict the diagnosis of reading disability?


# Use Cases:

1. look at relationships in the data between variables
    - API pulls data from RedCap
    - allows user to select multiple variables/features of interest
2. use machine learning to predict outcomes from variables in our data
    - performs dimentionality reduction and normalization of data
    - performs classification
    - builds model
    - uses model to estimate prediction
3. visualize results 
    - allows user to visualize relationships between chosen variables or interest interactivly from web application
    - allows visulaization of outcome predictions

