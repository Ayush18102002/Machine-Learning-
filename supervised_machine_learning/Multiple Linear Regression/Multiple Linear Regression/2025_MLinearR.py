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

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical data visualization
import sidetable  # For quick summary tables
from sklearn.compose import ColumnTransformer  # For column-wise transformations
from sklearn.pipeline import Pipeline  # For building pipelines
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.preprocessing import MinMaxScaler  # For scaling numerical features
from sklearn.preprocessing import OneHotEncoder  # For one-hot encoding categorical features
from feature_engine.outliers import Winsorizer  # For outlier treatment
from statsmodels.stats.outliers_influence import variance_inflation_factor  # For VIF calculation
from statsmodels.tools.tools import add_constant  # For adding constant to the model
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
import statsmodels.api as sm  # For statistical models and tests
from sklearn.linear_model import LinearRegression  # For linear regression modeling
from sklearn.metrics import r2_score  # For evaluating model performance
import joblib  # For saving and loading models
import pickle  # For serializing and deserializing Python objects
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV  # For cross-validation and hyperparameter tuning
from sklearn.feature_selection import RFE  # For recursive feature elimination
from sqlalchemy import create_engine  # For database connection
import dtale # AutoEDA library


from urllib.parse import quote

'''
Space (" ") and exclamation mark ("!") are NOT safe in a URL or to be used as password

from urllib.parse import quote

text = "This is a test!"
encoded_text = quote(text)
print(encoded_text)

# Space becomes %20 and ! becomes %21. Now it’s safe to put this into a URL or use as password!

'''

# Database connection
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "***",  # MySQL username
                               pw = quote("**********"),  # MySQL password
                               db = "cars_db"))  # MySQL database name 

# Load the offline data into Database to simulate client conditions
cars = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\Machine_learning\supervised_machine_learning\Multiple Linear Regression\Multiple Linear Regression\CarswithEnginetype.csv")
cars.to_sql('cars', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Read data from MySQL database
sql = 'SELECT * FROM cars'

dataset = pd.read_sql_query(sql, engine)  # Read data from SQL database using the provided SQL query and engine

# Separating input and output variables
X = pd.DataFrame(dataset.iloc[:, 1:6])  # Extract input features from the dataset
y = pd.DataFrame(dataset.iloc[:, 0])  # Extract output variable from the dataset

# Separating Non-Numeric features
categorical_features = X.select_dtypes(include = ['object']).columns  # Select non-numeric (categorical) features
print(categorical_features)  # Print the names of categorical features

# Separating Numeric features
numeric_features = X.select_dtypes(exclude = ['object']).columns  # Select numeric features
print(numeric_features)  # Print the names of numeric features

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# AutoEDA 
# Dtale
d = dtale.show(dataset)
d.open_browser()

# Or use any other AutoEDA library such as sweetviz
import sweetviz as sv
report = sv.analyze(cars)
report.show_html('sweetviz_report.html')

# Quick EDA Insights: 
# Based on the correlation matrix:
# Relationship between MPG, HP is polynomial in 2 degrees (quadratic). HP, HP*HP  
# Relationship between MPG, SP is polynomial in 2 degrees (quadratic). SP, SP*SP 
#  - VOL & WT: ~ 1 (very strong positive correlation)
#  - HP & SP : 0.973 (very strong positive correlation)
# High correlation suggests potential collinearity, which can negatively affect model accuracy.
# There are outliers in all numerical Xs
# There are missing value in all numerical Xs

# Define features
numeric_features = ["HP", "VOL", "SP", "WT"]
categorical_features = ["Enginetype"]

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

# Build column transformer (Numeric first, then Categorical — purely choice)
preprocess_pipeline = ColumnTransformer([
    ('numerical', num_pipeline, numeric_features),
    ('categorical', categ_pipeline, categorical_features)
])

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

# Add a constant term (intercept) to the clean data (assuming 'clean_data' is a DataFrame)
P = add_constant(Clean_data)

# Build a vanilla linear regression model (Ordinary Least Squares) using statsmodels
basemodel = sm.OLS(y, P).fit()  # 'y' is the target variable, 'P' is the data with constant term

# Summarize the model results
basemodel.summary()  # Print a summary of the model, including coefficients, p-values, R-squared, etc.

# High p-values of coefficients indicate insignificance due to collinearity
# Identify the variable with the highest collinearity using Variance Inflation Factor (VIF)
# Addressing Collinearity and Influential Observations

# Calculate Variance Inflation Factors (VIF)
vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
print(vif)  # Display VIF values for each feature

# Identify Feature with High VIF (assuming threshold is > 5 for high collinearity)
# Based on VIF values, a feature might have a high VIF (e.g., index 3).
# This suggests potential collinearity with other features.

# Drop Feature with Highest VIF WT = 96.932596
clean_data1 = Clean_data.drop('WT', axis = 1)  # Drop the feature with the highest VIF (assuming WT in this case)

# Build a Model on the Reduced Dataset
basemode2 = sm.OLS(y, clean_data1).fit()
basemode2.summary()  # Print the model summary for the reduced dataset


# If any of the coefficient values are insignificant then do the following (mentioned steps in comments)

'''
# Check for Influential Observations (Rows)
sm.graphics.influence_plot(basemode2)  # Create influence plots to identify potentially influential observations

# Handle Influential Observations (optional)
# Based on the influence plots, you might identify influential observations (e.g., index 76 and 78).
# These observations can potentially skew the model.

# Remove Influential Observations and Build Model on Updated Dataset
clean_data1_new = clean_data1.drop(clean_data1.index[[76, 78]])  # Drop identified influential observations
y_new = y.drop(y.index[[76, 78]])  # Drop corresponding target values
basemode3 = sm.OLS(y_new, clean_data1_new).fit()
basemode3.summary()  # Print the model summary for the updated dataset with influential observations removed

'''

# Perform transformations to capture the patterns better. 
clean_data2 = clean_data1.copy()

# Step 1: Add HP^2 to Cleaned Data
clean_data2['HP_squared'] = clean_data2['HP'] ** 2

# Build model with HP^2 included
model_with_hp2 = sm.OLS(y, clean_data2).fit()
print("\n=== Model Summary after adding HP^2 ===\n")
print(model_with_hp2.summary())


clean_data3 = clean_data1.copy()

# Step 2: Add SP^2 to Cleaned Data
clean_data3['SP_squared'] = clean_data3['SP'] ** 2

# Build model with HP^2 included
model_with_sp2 = sm.OLS(y, clean_data3).fit()
print("\n=== Model Summary after adding SP^2 ===\n")
print(model_with_sp2.summary())

clean_data4 = clean_data1.copy()

# Step 3: Add HP^2, SP^2 to Cleaned Data
clean_data4[['HP_squared', 'SP_squared']] = clean_data4[['HP', 'SP']] ** 2

# Build model with HP^2, SP^2 included
model_with_hp2_sp2 = sm.OLS(y, clean_data4).fit()
print("\n=== Model Summary after adding HP^2 & SP^2 ===\n")
print(model_with_hp2_sp2.summary())

clean_data5 = clean_data1.copy()

clean_data5[['HP_squared', 'VOL_squared', 'SP_squared']] = clean_data5[['HP', 'VOL', 'SP']] ** 2

# Build model with HP^2, VOL^2, SP^2 included
model_with_hp2_vol2_sp2 = sm.OLS(y, clean_data5).fit()
print("\n=== Model Summary after adding HP^2, VOL^2 & SP^2 ===\n")
print(model_with_hp2_vol2_sp2.summary())

# Split Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(clean_data5, y, test_size = 0.2, random_state = 0)

# Build the Final Model (without cross-validation for simplicity)
model = sm.OLS(Y_train, X_train).fit()
model.summary()  # Print the model summary for the final model

# Evaluate Model Performance on Training and Testing Data (without Cross-Validation)

# Training Data Performance
ytrain_pred = model.predict(X_train)  # Predict target values for training data using the fitted model

from sklearn.metrics import mean_absolute_percentage_error
# MAPE for Testing Data
mape_train = mean_absolute_percentage_error(Y_train, ytrain_pred) * 100
print(f"Training MAPE: {mape_train:.2f}%")

# Testing Data Performance
ytest_pred = model.predict(X_test)  # Predict target values for testing data using the fitted model

mape_test = mean_absolute_percentage_error(Y_test, ytest_pred) * 100
print(f"Testing MAPE: {mape_test:.2f}%")

# Cross-Validation for Model Selection and Evaluation (using KFold)
# When  we have a tiny dataset, it is better to train on every fold and test on 
# every fold and this increases the confidence with which you communicate the 
# accuracy values to wider stakeholders. 
lm = LinearRegression()  # Create a linear regression model object
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)  # Define KFold cross-validation with 5 splits, shuffling, and fixed random state
scores = cross_val_score(lm, clean_data5, y, scoring = 'r2', cv = folds)  # Perform KFold cross-validation to get R-squared scores on each fold
scores

# Calculate average cross-validation R2 score
overall_cv_score = scores.mean()
print(f"Overall Cross-Validation R2 Score: {overall_cv_score:.4f}")

std_cv_score = scores.std()
print(f"Standard Deviation of CV Scores: {std_cv_score:.4f}")

# Saving the Best Model
pickle.dump(model_with_hp2_vol2_sp2, open('finalmodel.pkl', 'wb')) 

# END OF CODE