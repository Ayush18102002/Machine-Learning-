'''
CRISP-ML(Q) framework has six phases:

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building + Hyperparameter Optimization / Tuning
- Evaluation - Business, ML & Economic Success Criteria
- Deployment - On-premise or Cloud and should be put the model in container (Docker)
- Monitoring and Maintenance

Business Problem: 
        In mileage-sensitive markets, automobile manufacturers face the challenge of accurately estimating 
        the fuel efficiency of new car designs. Without robust modeling, manufacturers must rely on costly 
        experiments and guesswork, potentially leading to inefficient production decisions, unmet consumer 
        expectations, and reduced profitability.

High-Level Solution
        Develop and deploy a predictive analytics model that leverages relevant vehicle parameters—such 
        as engine specifications, design features, and historical performance data—to reliably estimate 
        a new car’s mileage before mass production. This empowers manufacturers to optimize fuel efficiency 
        and cost-effectively meet market demands.

Business Objective(s): 
        Maximize new-car fuel efficiency
        Minimize engineering cost overhead

Business Constraint(s):
        Minimize substandard mileage variants

Success Criteria - Must be quantitative / SMART
        Business: Achieve at least a 15% increase in average fuel efficiency for newly launched models 
                  compared to previous generations.
        ML:       MAPE < 15% when predicting mileage on the validation dataset.
        Economic: Realize a 10% cost reduction in overall manufacturing/engineering expenses tied to 
                  mileage improvements.

HLD - DAR - DLD (Watch "AI for Business Professionals" module in LMS)
Create Project Charter - Contains high level information of the project. 

1.b. Data Understanding
# Data Collection: Data Sources -> Data Storage -> EDA (Business & Statistical Insights)
    Data: (Meta data details are needed)

MPG             Mileage of the car              Miles per Gallon (mpg)
Enginetype      Type of engine used in the car (e.g., petrol, diesel, hybrid, lpg)
HP              Horsepower of the engine        Horsepower (HP)
VOL             Engine volume/displacement      Cubic Inches (in³) or Cubic Centimeters (cc) (unit to be clarified)
SP              Top speed of the car            Miles per Hour (mph)
WT              Weight of the car               Tons
'''

# Load the Data into Python and perform EDA and Data Preprocessing

# Code Modularity is important

# Importing necessary libraries
import pandas as pd  # for dataframes and data analysis
import numpy as np  # for numerical operations and arrays

# Import libraries for data visualization
import matplotlib.pyplot as plt  # for creating plots and charts

# Import library for printing fancy data tables (optional)
import sidetable  # for creating visually appealing data tables

# Import libraries for building and evaluating machine learning models
import statsmodels.api as sm  # Library for statistical modeling and testing
from sklearn.linear_model import Ridge, Lasso, ElasticNet  # Models for Ridge, Lasso, and ElasticNet regression
from sklearn.compose import ColumnTransformer  # for combining preprocessing steps
from sklearn.pipeline import Pipeline  # for chaining data processing steps
from sklearn.impute import SimpleImputer  # for handling missing data
from sklearn.preprocessing import MinMaxScaler  # for scaling numerical features
from sklearn.preprocessing import OneHotEncoder  # for encoding categorical features
from feature_engine.outliers import Winsorizer  # for capping outliers (optional)

# Import libraries for statistical analysis
from statsmodels.tools.tools import add_constant  # for adding a constant term to models
from statsmodels.stats.outliers_influence import variance_inflation_factor  # for detecting multicollinearity

# Import libraries for model persistence
import joblib  # for saving and loading scikit-learn models (recommended)
import pickle  # for saving and loading Python objects (less secure than joblib)

# Import library for hyperparameter tuning (optional)
from sklearn.model_selection import GridSearchCV  # for grid search cross-validation

# Import library for database connection (optional)
from sqlalchemy import create_engine  # for connecting to databases
from urllib.parse import quote

# Create a connection engine to the MySQL database named 'cars_db'
# using credentials 'user1' and 'user1' (replace with your actual credentials)
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                        .format(user = "root", pw = quote("Ayush.2002@#"), db = "cars_db"))

# Read the 'cars' table from the MySQL database into a Pandas DataFrame
sql = 'SELECT * FROM cars'
dataset = pd.read_sql_query(sql, engine)

# Display summary statistics of the dataset's numerical columns
dataset.describe()
dataset.info()

# Check for missing values in each column (True indicates missing values)
print(dataset.isnull().any())

# Display data types and non-null counts for each column
dataset.info()

# Separate the feature matrix (X) containing all columns except the first
# (assuming the first column is the target variable)
X = dataset.iloc[:, 1:6]  # Select all rows and columns from 1 (inclusive) to 6 (exclusive)

# Create the target variable DataFrame (y) containing the first column
y = dataset.iloc[:, 0]  # Select all rows and the first column (0)

# Explore unique values and counts for the categorical feature "Enginetype"
print(X["Enginetype"].unique())  # Display unique engine types
print(X["Enginetype"].value_counts())  # Count occurrences of each engine type

# Explore frequencies of all categorical variables using sidetable (optional)
X.stb.freq(["Enginetype"])  # Display frequencies of all categorical features (if sidetable is imported)

# Separate categorical and numerical features based on data types
categorical_features = X.select_dtypes(include = ['object']).columns  # Get categorical column names
numeric_features = X.select_dtypes(exclude = ['object']).columns     # Get numerical column names

## Data Preprocessing
# Imputation strategy for numeric columns
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'mean')),    # Step 1: Fill missing values
    ('winsorize', Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5)),  # Step 2: Handle outliers
    ('scale', MinMaxScaler())  # Step 3: Scale the cleaned data
])

# Encoding categorical to numeric variable
categ_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('label', OneHotEncoder(sparse_output = False, drop = 'first', handle_unknown = 'ignore')) # handle_unknown = 'ignore', even if a new unseen category comes during prediction, it won't crash. It'll just ignore it. 
])

# Using ColumnTransfer to transform the columns of an array or pandas DataFrame. This estimator allows different columns or column subsets of the input to be transformed separately and the features generated by each transformer will be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('numerical', num_pipeline, numeric_features), 
                                         ('categorical', categ_pipeline, categorical_features)])

# Apply preprocessing
preprocess_pipeline.set_output(transform = "pandas")
Clean_data = preprocess_pipeline.fit_transform(X)

# After transformation
final_feature_names = preprocess_pipeline.get_feature_names_out()
print("All transformed feature names:")
print(final_feature_names)

# Save the pipeline
joblib.dump(preprocess_pipeline, 'preprocessed_pipeline.pkl')

import os
os.getcwd()

# Rename columns to remove prefixes (numerical__ and categorical__)
Clean_data.columns = [col.split("__")[-1] for col in Clean_data.columns]

# Display cleaned column names
print("Renamed Feature Names:", Clean_data.columns.tolist())

Clean_data.info()  # Display information about the cleaned data DataFrame
eda_post_preprocessing = Clean_data.describe()
print(eda_post_preprocessing)

# Save the clean data in the database in a different table. 
# This table will be called as Feature Store and will be used for model building phase. 

# Model Building Phase of CRISP-ML(Q) #
# Build a Baseline Linear Regression Model

# Create a constant term (usually for intercept) and add it to the cleaned data
# for use with statsmodels' OLS model
P = add_constant(Clean_data)

# Fit a statsmodels Ordinary Least Squares (OLS) model on the combined data (P) and target variable (y)
basemodel = sm.OLS(y, P).fit()  # 'sm' likely refers to 'statsmodels'
basemodel.summary()

# Check for multicollinearity using Variance Inflation Factor (VIF)
vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])],
                index = P.columns)  # Calculate VIF for each feature
vif

# Identify features with high VIF (potential multicollinearity)
# You may need to adjust the threshold based on your domain knowledge and tolerance
high_vif_cols = vif[vif > 5]  # Example: Consider features with VIF > 5 to be highly correlated
print("Features with potentially high multicollinearity (VIF > 5):", high_vif_cols.index.tolist())

# Create a new DataFrame 'clean_data1' by dropping a feature with high VIF (optional)
# Consider using domain knowledge and feature importance along with VIF to decide which feature to remove
clean_data1 = P.drop('WT', axis = 1)  # Example: Drop the 'WT' feature (replace with the actual high VIF feature)

# Refit the OLS model on the data without the potentially problematic feature
basemodel2 = sm.OLS(y, clean_data1).fit()
basemodel2.summary()

# Hyperparameter tuning for Lasso, Ridge, and ElasticNet regression
lasso = Lasso()  # Initialize Lasso regression model

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.13, 1, 5, 10, 20]}  # Define alpha values for hyperparameter tuning

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)  # Perform grid search with cross-validation to find best alpha for Lasso

lasso_reg.fit(clean_data1, y)  # Fit Lasso regression model to the cleaned data

lasso_pred = lasso_reg.predict(clean_data1)  # Make predictions using the trained Lasso regression model



ridge = Ridge()  # Initialize Ridge regression model

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)  # Perform grid search with cross-validation to find best alpha for Ridge

ridge_reg.fit(clean_data1, y)  # Fit Ridge regression model to the cleaned data

ridge_pred = ridge_reg.predict(clean_data1)  # Make predictions using the trained Ridge regression model



enet = ElasticNet()  # Initialize ElasticNet regression model

enet_reg = GridSearchCV(enet, parameters, scoring = 'r2', cv = 5)  # Perform grid search with cross-validation to find best alpha for ElasticNet

enet_reg.fit(clean_data1, y)  # Fit ElasticNet regression model to the cleaned data

enet_pred = enet_reg.predict(clean_data1)  # Make predictions using the trained ElasticNet regression model


# Compare scores of different regression models
scores_all = pd.DataFrame({'models': ['Lasso', 'Ridge', 'Elasticnet'],
                           'Scores': [lasso_reg.best_score_, ridge_reg.best_score_, enet_reg.best_score_]})

# Save the best model
# Extract the best ElasticNet model from the GridSearchCV object
finalgrid = enet_reg.best_estimator_

# Save the best ElasticNet model for later use
pickle.dump(finalgrid, open('grid_elasticnet.pkl', 'wb'))  # 'wb' for binary write

# Quick Dip Test before final deployment
# Load the best ElasticNet model for prediction
model1 = pickle.load(open('grid_elasticnet.pkl', 'rb'))  # 'rb' for binary read

# Load the previously saved preprocessing models
clean = joblib.load('preprocessed_pipeline.pkl')

# Read the test data from an Excel file
data = pd.read_excel(r"C:\Users\admin\OneDrive\Desktop\Machine_learning\supervised_machine_learning\Lasso, Ridge, Elastic Net\carswithenginetype_test.xlsx")

# Preprocess the test data
clean_data = pd.DataFrame(clean.transform(data), columns = clean.get_feature_names_out())

# Rename columns to remove prefixes (numerical__ and categorical__)
clean_data.columns = [col.split("__")[-1] for col in clean_data.columns]

# Display cleaned column names
print("Renamed Feature Names:", clean_data.columns.tolist())

P = add_constant(clean_data)

clean_data1 = P.drop('WT', axis = 1)

# Make predictions on the preprocessed test data using the best ElasticNet model
prediction = pd.DataFrame(model1.predict(clean_data1), columns = ['PredictedValues_MPG'])

# Combine predictions with the original test data for analysis
final = pd.concat([prediction, data], axis = 1)

# Display the final DataFrame containing predictions and original data
print(final)

# END OF CODE
