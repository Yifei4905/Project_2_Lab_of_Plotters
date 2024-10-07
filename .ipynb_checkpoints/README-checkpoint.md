# Week 17 Kala Project 2 Notebook - Census Income Prediction

## Developing an ML model to solve, analyze, or visualize a problem of your choice

### Data Source - https://archive.ics.uci.edu/dataset/117/census+income+kdd

### Questions for exploration
1. Is there a relationship between census demographic data and income over/under USD 50,000?
1. Which is the best model to predict income over/under 50K?
1. Can the performance of the models be explained?

### Step 1: Preprocessing of data
* Used Standard Scalar to scale data.
* Used category encoding for categorical data.
* Input data is very skewed, only 6% of rows show >50K USD (1), 94% of data is <50K USD (0)

### Step 2: Evaluations of ML models
#### I tried the following models:
1. Logistic Regression
1. Random Forest
1. Support Vector Classifier
1. Decision Tree
1. Gradient Boosting
1. ADA Boosting
1. Extra Trees

![alt text](Model_Stats.png)

### Step 3: Exploratory Data Analysis
* I tried to cluster the data using Principal Component Analysis (PCA).
* Using 2 components only explained variance of 26.76%, adding additional components did not increase the explained variance by much.
* K-Means displayed an elbow curve at 3 or 4.
* While the clusters of 2, 3 or 4 look pretty distinct, I am not sure of the usability of this analysis.
* I tried both over and undersampling, but it did not change the PCA by much.

![alt text](PCA_Picture.png)

### Conclusions:
* In general, Random Forest model had the best F1 scores (54%-98%)
* Gradient Boost model was a close second at 53%-98%
* Undersampled Random Forest model had the best balanced accuracy score (87%)
* PCA analysis was not effective at reducing dimensionality
* The balanced accuracy score seems to come at the expense of the F1 scores
* For a better analysis, I could get more balanced data directly from https://data.census.gov/.