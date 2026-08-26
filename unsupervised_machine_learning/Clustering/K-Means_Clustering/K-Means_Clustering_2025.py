# K-Means Clustering

'''
# `CRISP-ML(Q)` process model describes six phases:

# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance

# Business Problem:
    Students have to evaluate a lot of factors before taking a decision 
    to join a university for their higher education requirements.

# High Level Solution:
    Logically grouping the available universities will allow understanding the characteristics of each group.

# Objective(s): Maximize the convenience of the admission process
# Constraint(s): Minimize the brain drain

Success Criteria

# Business Success Criteria: Reduce the application process time from anywhere between 20% to 40%
# ML Success Criteria: Achieve Silhouette coefficient of at least 0.6
# Economic Success Criteria: US Higher education department will see an increase in revenues by at least 30%

# HLD - DAR - DLD (Data Pipeline & Model Pipeline)
 
'''
# Data Understanding: 
# Data Sources - Data Collection - Data Storage - EDA

# Data: 
#    The university details are obtained from the US Higher Education Body and is publicly available for students to access.
# 
# Data Dictionary:
# - Dataset contains 25 university details
# - 7 features are recorded for each university
# 
# Meta Data Description: (Features, Description of features, Units of measure, Values within each feature)
# - Univ - University Name
# - State - Location (state) of the university
# - SAT - Cutoff SAT score for eligibility
# - Top10 - % of students who ranked in the top 10 in their previous academics
# - Accept - % of students admitted to the universities
# - SFRatio - Student to Faculty ratio
# - Expenses - Overall cost in USD
# - GradRate - % of students who graduate


# Code modularity


# #### Install the required packages if not available
# !pip install feature_engine
# !pip install sklearn_pandas

# **Importing required packages**
# import numpy as np
import pandas as pd  # Importing pandas library for data manipulation and analysis
#import sweetviz  # Importing sweetviz for automated exploratory data analysis
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

from sklearn.pipeline import Pipeline  # Importing Pipeline for chaining preprocessing steps
from sklearn.impute import SimpleImputer  # Importing SimpleImputer for handling missing values
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for feature scaling
from sklearn.preprocessing import OrdinalEncoder # Importing OrdinalEncoder for converting string to integer
from sklearn.compose import ColumnTransformer # Importing ColumnTransformer to transfer pipelines into the data 

from sklearn.cluster import KMeans  # Importing KMeans for clustering
from sklearn import metrics  # Importing metrics for evaluating clustering performance
import joblib  # Importing joblib for saving trained models
import pickle  # Importing pickle for saving Python objects
from sqlalchemy import create_engine, text
from urllib.parse import quote
# Importing the data from an Excel file
uni = pd.read_excel(r"C:\Users\admin\OneDrive\Desktop\python-practice\K-Means_Clustering\University.xlsx")

# Credentials to connect to the database
user = 'root'  # Username
pw = quote('*****************')  # Password
db = 'univ_db'  # Database name

# Creating a database engine to connect to the MySQL database using the provided credentials
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Using the to_sql() function to push the DataFrame 'uni' onto a SQL table named 'univ_tbl' in the database
# The 'if_exists' parameter is set to 'replace' to replace the table if it already exists
# The 'chunksize' parameter specifies the number of rows to write at a time
# The 'index' parameter is set to False to avoid writing row indices to the SQL table
uni.to_sql('univ_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Defining a SQL query to select all records from the 'univ_tbl' table
sql = text('select * from univ_tbl;')

# Executing the SQL query and reading the results into a DataFrame 'df' using read_sql_query() function
df = pd.read_sql_query(sql, engine.connect())

# Displaying the data types and non-null counts of each column in the DataFrame 'df'
df.info()

# Dropping the unwanted features "UnivID" and "Univ" from the DataFrame 'df' and creating a new DataFrame 'df1'
df1 = df.drop(["UnivID", "Univ"], axis = 1)

# Displaying the first few rows of the DataFrame 'df1' after dropping the unwanted features
df1.head()


# # EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# ***Descriptive Statistics and Data Distribution Function***
# Generating descriptive statistics of the DataFrame 'df1', including count, mean, standard deviation, minimum, maximum, and quartile values for numerical columns
df1.describe()

# Checking unique values for the categorical feature 'State' in the DataFrame 'df1'
# The unique() method returns an array of unique values
unique_states = df1.State.unique()
unique_states

# Counting the number of unique states
num_unique_states = df1.State.unique().size
num_unique_states

# Counting the occurrences of each unique state and displaying them in descending order
state_value_counts = df1.State.value_counts()
state_value_counts




# AutoEDA
# Automated Libraries
# D-Tale
########

#pip install dtale
import dtale

d = dtale.show(df, host = 'localhost', port = 8000)
d.open_browser()


# Checking for missing data in the DataFrame 'df1'
# The isnull() method returns a DataFrame of boolean values indicating whether each element is missing
# The sum() method sums up the missing values for each column
# This provides the count of missing values in each column
missing_data = df1.isnull().sum()

# Segregate Numeric and Non-numeric columns
df1.info()

# Selecting numeric columns (excluding object dtype) from the DataFrame 'df1' and storing their column names in 'numeric_features'
numeric_features = df1.select_dtypes(exclude = ['object']).columns

# Displaying the numeric features
numeric_features

# Selecting non-numeric columns (object dtype) from the DataFrame 'df1' and storing their column names in 'categorical_features'
categorical_features = df1.select_dtypes(include = ['object']).columns

# Displaying the non-numeric features
categorical_features

# Defining a Pipeline to deal with missing data and scaling numeric columns
# The Pipeline consists of two steps: imputation using mean strategy and scaling using MinMaxScaler
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean')), ('scale', MinMaxScaler())])

# Displaying the defined pipeline
num_pipeline

# Encoding Non-numeric fields
# Defining a pipeline to convert categorical data into numeric data 
categ_pipeline = Pipeline([('OrdinalEncoding', OrdinalEncoder())])

# Displaying the defined pipeline
categ_pipeline

# Using ColumnTransfer to transform the Pipelines into the data. 
# This estimator allows different columns or column subsets of the input to be
# transformed separately and the features generated by each transformer will
# be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([ ('categorical',categ_pipeline, categorical_features),
                                       ('numerical', num_pipeline, numeric_features)], 
                                        remainder = 'passthrough') # Skips the transformations for remaining columns

preprocess_pipeline

# Pass the raw data through pipeline
processed = preprocess_pipeline.fit(df1) 


# ## Save the Imputation and Encoding pipeline
# import joblib
joblib.dump(processed, 'preprocessing')

# File gets saved under current working directory
import os
os.getcwd()

# Clean and processed data for Clustering
univ_clean = pd.DataFrame(processed.transform(df1), columns = processed.get_feature_names_out())
univ_clean

#Saving preprocessed data

univ_clean.to_sql('univ_clean', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Clean data
univ_clean.describe()

# CLUSTERING MODEL BUILDING

# KMeans Clustering 

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#pip install kneed
from kneed import KneeLocator 
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote
import pickle
import os

# Define the range of K values (number of clusters)
k_values = list(range(2, 9))

# Randomly select 5 values of K to evaluate
random_k_values = random.sample(k_values,5)

# Store results for finding the best K
best_k = None  # To store the best K value
best_score = -1  # Initialize best silhouette score
TWSS = []  # Total Within-Cluster Sum of Squares

# Loop through each randomly selected K
for k in random_k_values:
    kmeans = KMeans(n_clusters = k, init = 'random', random_state = 42)
    labels = kmeans.fit_predict(univ_clean)
    
    # Compute silhouette score (higher is better)
    score = silhouette_score(univ_clean, labels) if len(set(labels)) > 1 else -1

    # Store TWSS (inertia) for scree plot
    TWSS.append((k, kmeans.inertia_))

    # Update best K if the score improves
    if score > best_score:
        best_score = score
        best_k = k

# Print the best K value found
print("Best K:", best_k)
print("Best Silhouette Score:", best_score)

# Convert TWSS list to sorted format for plotting
TWSS.sort()
k_values_sorted, inertia_values = zip(*TWSS)

# Scree plot (Elbow Method) for choosing K visually
plt.plot(k_values_sorted, inertia_values, 'ro-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Total Within-Cluster Sum of Squares (TWSS)")
plt.title("Elbow Method Scree Plot")
plt.show()

# --------------------------
# Using KneeLocator for Best K Detection
# --------------------------

inertia_list = []  # To store inertia values

# Loop through the full K range for KneeLocator
for k in range(2, 9):
    kmeans = KMeans(n_clusters = k, init = "k-means++", random_state = 42)
    kmeans.fit(univ_clean)
    inertia_list.append(kmeans.inertia_)

# Find the best K using KneeLocator
kl = KneeLocator(range(2, 9), inertia_list, curve = 'convex', direction = 'decreasing')
best_k_knee = kl.elbow  # Optimal K determined by the elbow point

# Print best K detected by KneeLocator
print("Best K (Knee Method):", best_k_knee)

# Plot Knee Method for visualizing the elbow point
plt.style.use("ggplot")
plt.plot(range(2, 9), inertia_list, marker='o', linestyle='-')
plt.xticks(range(2, 9))
plt.ylabel("Inertia")
plt.xlabel("Number of Clusters (K)")
plt.axvline(x=best_k_knee, color='r', linestyle='--', label=f'Elbow at K={best_k_knee}')
plt.legend()
plt.title("Knee Method for Optimal K")
plt.show()

# --------------------------
# Final KMeans Model with Optimal K
# --------------------------

# Set the final number of clusters to the best found K
final_k = best_k_knee if best_k_knee else best_k  # Prioritize KneeLocator result

# Create KMeans model with best K
final_model = KMeans(n_clusters = final_k, init = "k-means++", random_state = 42)

# Fit the model
final_model.fit(univ_clean)

# Get cluster labels
cluster_labels = final_model.labels_

# Print final cluster assignments
print("Final Cluster Labels:", np.unique(cluster_labels))

# --------------------------
# Cluster Evaluation Metrics
# --------------------------
from sklearn import metrics

# Silhouette Score: Measures cohesion and separation
silhouette_score_value = metrics.silhouette_score(univ_clean, final_model.labels_)
print("Silhouette Score:", silhouette_score_value)

# Calinski-Harabasz Score: Measures cluster separation
calinski_harabasz_score = metrics.calinski_harabasz_score(univ_clean, final_model.labels_)
print("Calinski-Harabasz Score:", calinski_harabasz_score)

# Davies-Bouldin Index: Lower values indicate better clustering
davies_bouldin_score = metrics.davies_bouldin_score(univ_clean, final_model.labels_)
print("Davies-Bouldin Score:", davies_bouldin_score)

# --------------------------
# Saving Model using Pickle
# --------------------------
pickle.dump(final_model, open('Clust_Univ.pkl', 'wb'))
print("Model saved successfully.")

# --------------------------
# Exporting Results
# --------------------------

# Obtaining cluster labels as a Pandas Series
mb = pd.Series(cluster_labels)

# Concatenating cluster labels with original data
df_clust = pd.concat([mb, df.Univ, df1], axis = 1)
df_clust = df_clust.rename(columns = {0: 'cluster_id'})

# Display first few rows of the clustered data
print(df_clust.head())

# Aggregate data using mean for each cluster
cluster_agg = df_clust.iloc[:, 3:].groupby(df_clust.cluster_id).mean()
print(cluster_agg)

# Save results to CSV
df_clust.to_csv('KMeans_University.csv', encoding = 'utf-8', index = False)
print("Results saved to CSV.")

# Print current working directory
print("Current Directory:", os.getcwd())

# --------------------------
# Saving Results to MySQL Database
# --------------------------
df_clust.to_sql('univ_final_results', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
print("Data saved to MySQL successfully.")

# End of KMeans Clustering
