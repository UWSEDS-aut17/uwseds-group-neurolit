## Functional Specification

# User Profile:

- Our users will be research scientists familiar with python, and specifically, Jupyter Notebook.
- Our users will be familiar with the various reading measures, aptitude tests, and brain imaging data (MEG).
- Our users will also be familiar with the various survey questions used to create the survey dataset.

# Elements of the problem statement:

- What survey measures are indicative of reading skill?
- Do MEG measures predict reading performance/math performance/or other cognitive skills?
- What are the predictors of a diagnosis of a learning disability?
- Can we use machine learning to predict reading outcomes, discover patterns, and/or reveal relationships between survey responses and academic skills?
- Do other experiences (media use/music education) explain variance in reading skill?
- Can we use reading outcomes to predict experience (ex. Homeschooling, media use, experience with text)?
- API that allows users to curate data, analyze it using machine learning techniques, and visualize the data and results in intuitive ways.


# Use Cases:

1. Looking at the role of parent perceptions of child’s reading ability to predict reading performance (example of *DATA CURATION* use case)
- User will select survey variables of interest
  - Challenge: how will the variable of interest be selected?
    - There will need to be ‘library’ of variables
    - Will the user be able to select multiple variables?
- API will pull data from RedCap
  - Challenge: How will the program interact with the dataset?
    - The data will be constantly changing and growing
    - Will the user need to download the most recent data?
    - Will the API interact directly with the database?
- API will output visualization of relationship with reading performance
  - Challenge: How will reading performance be assessed?
    - Will each reading score be assessed independently?
    - WIll we determine specific reading scores of interest as well?
    - Will a composite measure be computed?
  - Challenge: How will the results be visualized?
    - Will there be an interaction figure? Bokeh?
- API will determine the ability of the prescribed variable to explain variance in performance
  - Challenge: How will the API compute this statistic?
  - Will the API regress? Or will the observation of patterns be the optimal output?

2. Does brain data predict reading or math performance?
- User will decide which brain and reading/math measures to compare
  - Challenge: What components of the brain data will we use in the machine learning algorithm? (ex: Temporal, spatial, frequency)
    - Selecting a specific MEG component measure which would be used to correlate with behavioural scores
    - There will have to be a limit on number of components
- Visualization of results
  - Challenge: How to visualize results?
    - Choose existing functions/packages can we use (mne-python, matplotlib)
- Selecting ways to visualize (plots, graphs)
  - Visualization could be interactive
  - Challenge: what metric do we return? (plots, numbers etc)

3. Trying to predict whether a kid has been homeschooled based on hobbies
- User will need to select variables corresponding to hobbies
  - Challenge: Can we provide meta-categories for our data that makes it easier for users to select just hobbies-focused features?
- Selecting the feature corresponding to whether or not a kid has been homeschooled
- Splitting up the data into train and test sets at random
- User will want to use a specific set of algorithms on training data
  - Need functionality for dimensionality reduction techniques and supervised learning algorithms for small-scale data (SVMs, k-nn, decision trees, etc.)
- Metrics for prediction accuracy on training dataset
- Evaluate prediction accuracy on test dataset
- Visualizing relationships between predictor variables and outcomes
  - Colored scatter plots
  - Heatmaps
  - Histograms
  - etc.


# Curating Data:

- Figure out how to use RedCap Python API
- Create a meta-label CSV file
- Users should be able to select features individually, select features based on meta-labels, drop individual features, drop features based on meta-labels, print existing features, print available features, print meta-label features
- Create a variable type CSV file
- Look at univariate distributions
- Look at joint distributions

# Analysing Data:
