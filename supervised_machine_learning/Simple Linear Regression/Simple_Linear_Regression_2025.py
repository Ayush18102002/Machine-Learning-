''' Simple Linear regression
Simple linear regression is a regression model that estimates the relationship
between one independent variable (X) and a dependent variable (Y) using a straight line.

# CRISP-ML(Q) process model describes six phases:
- Business & Data Understanding
- Data Preparation
- Model Building (Machine Learning) & HPO
- Evaluation
- Deployment
- Monitoring & Maintenance

1.a. Business Understanding
Problem Statement
    # Studies have shown that individuals with excess Adipose tissue (AT) in 
    # their abdominal region have a higher risk of cardiovascular diseases.
    # To assess the health conditions of a patient, doctor must get a report 
    # on the patients AT values. Computed Tomography, commonly called the CT Scan
    # is the only technique that allows for the precise and reliable measurement 
    # of the AT (at any site in the body). 

# The problems with using the CT scan are:
    - Many physicians do not have access to this technology
    - Irradiation of the patient (suppresses the immune system)
    - Expensive

# The Hospital/Organization wants to find an alternative solution for this 
# problem, which can allow doctors to help their patients efficiently.

# Objective(s):  Minimize patient exposure to radiation
                 Minimize dependency on CT scans
# Constraint(s): Minimize the need for specialized equipment
                 Maximize clinical interpretability of results
                 
# Research: A group of researchers conducted a study with the aim of predicting 
            abdominal AT area using simple anthropometric measurements, i.e., 
            measurements on the human body.
 
# High Level Proposed Plan / Solution:
  The Waist Circumference – Adipose Tissue data should be part of this study, wherein
  the aim is to study how well waist circumference (WC) predicts the AT area.

# Success Criteria (Should be quantitative - SMART)
    Business: 
            >80% physician adoption within 6 months
            ≥90% reduction in CT scan usage for AT measurement

    Machine Learning: 
            R² ≥ 0.85 in predicting AT from alternative measurements (e.g., waist circumference, BMI, age, gender)
            MAPE ≤ 5% of predicted AT values compared to CT-derived ground truth 

    Economic Success: 
            ≥60% cost reduction in AT diagnosis workflow
            Break-even ROI within 1 year of deployment

HLD - DAR - DLD (Watch "AI for Business Professionals" module in LMS)
Create Project Charter - Contains high level information of the project. 

1.b. Data Understanding
# Data Collection: Data Sources -> Data Storage -> EDA (Business & Statistical Insights)
    Data: (Meta data details are needed)
     Y = Dependent variable = AT values from the historical Data - square centimeters (cm²).
     X = Independent variable = Waist Circumference of these patients - centimeters (cm).

1. Evaluate the available Hospital records for relevant data (CT scan of patients) - Secondary Data
 
2. Record the Waist Circumference of patients - Primary Data

- Strategy to Collection Primary Data:
    Call the most recent patients (1 week old) with an offer of free 
    consultation from a senior doctor to attract them to visit hospital.
    Once the patients visit the hospital, we can record their 'Waist 
    Circumference' using measuring tape.
'''

# Code modurality (arranging blocks of relevant code together)

# Explore the Patients Database (MySQL)

# Importing relevant libraries/packages
import pandas as pd
from urllib.parse import quote
from sqlalchemy import create_engine # Import the library for connecting to databases
from sqlalchemy import text # use for SQL queries in Python
import dtale # AutoEDA library
# Import necessary libraries for data manipulation, mathematical calculations, visualization, and modeling
import pandas as pd  # Data Manipulation
import numpy as np   # Mathematical calculations
import matplotlib.pyplot as plt  # Data Visualization
import seaborn as sns  # Data Visualization
import joblib  # Saving and loading model
import pickle  # Saving and loading model
from sklearn.compose import ColumnTransformer  # Column Transformer
from sklearn.pipeline import Pipeline  # Pipeline for modeling
from sklearn.impute import SimpleImputer  # Imputing missing values
from sklearn.preprocessing import StandardScaler # Feature scaling
from sklearn.pipeline import make_pipeline  # Pipeline for modeling
from feature_engine.outliers import Winsorizer  # Handling outliers
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
import statsmodels.formula.api as smf  # Statsmodels for statistical modeling
from sklearn.preprocessing import PolynomialFeatures  # Polynomial features for modeling
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.model_selection import GridSearchCV # Hyperparameter Tuning

# Load the patient AT data
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\Machine_learning\supervised_machine_learning\Simple Linear Regression (1)\ATpatients.csv")

# Load the patient waist data
data2 = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\Machine_learning\supervised_machine_learning\Simple Linear Regression (1)\waist.csv")

# 1. Establish a connection to a MySQL database:
# Create an engine object, providing authentication details and database name.
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                     .format(user = "*****",  # Database username
                             pw = quote("***********"),  # Database password
                             db = "wcat"))  # Database name

# 2. Write data to separate tables:
#   - Write patient AT data to a table named 'atpatients'.
#      - 'if_exists = 'replace': Overwrites the table if it already exists.
#      - 'index = False': Prevents writing the DataFrame index as a separate column.
data.to_sql('atpatients', con = engine, if_exists = 'replace', index = False)

#   - Write patient waist data to a table named 'waist'.
data2.to_sql('waist', con = engine, if_exists = 'replace', index = False)

# 3. Set primary keys:
#   - Define a primary key in both tables to ensure data integrity and efficient retrieval.

with engine.connect() as con:
    con.execute(text("ALTER TABLE atpatients ADD PRIMARY KEY (Patient);")) # Primary key for 'atpatients' table

with engine.connect() as con:
   con.execute(text("ALTER TABLE waist ADD PRIMARY KEY (Patient);"))  # Primary key for 'waist' table

# 4. Retrieve selected data:
#   - Construct an SQL query to fetch only necessary features from both tables, joining them on the 'Patient' column.
sql = "SELECT A.Patient, A.AT, A.Sex, A.Age, B.Waist from atpatients as A Inner join waist as B on A.Patient = B.Patient;"

#   - Execute the query and store the results in a pandas DataFrame for further processing.
wcat_full = pd.read_sql_query(sql, engine)

# Display basic information about the DataFrame
# Analyze and prepare the data for regression modeling

# 1. Get basic information about the data (wcat_full)
wcat_full.info()  # Print information like data types, number of non-null values, etc. for each column
wcat_full.describe()

# 2. Data Cleaning (Feature Selection):
#    - Create a new DataFrame (wcat) by dropping irrelevant features for your regression analysis.
#    - Dropped features here are 'Patient', 'Sex', and 'Age' based on your problem definition.
wcat = wcat_full.drop(["Patient", "Sex", "Age"], axis = 1)

# 3. Analyze the DataFrame after Dropping Features (wcat)
wcat.info()  # Print information about the DataFrame 'wcat' after dropping features

# Note: Depending on your specific analysis, you might need to perform additional data cleaning steps like handling missing values or outliers.

# Manual EDA
# Exploratory Data Analysis (EDA) - Get basic information about the data
# Run both lines of code together
print("Summary of the data:")
wcat.describe()  # Provides statistics like mean, standard deviation, etc. for each column

# View the first 10 rows of the data # Run both lines of code together
print("First 10 rows of the data:")
wcat.head(10)

# Sort data by waist circumference (ascending order)
wcat.sort_values('Waist', ascending = True, inplace = True)  # Sorts the data available in RAM (memory)
# Reset the index after sorting (optional, but keeps indexing clean)
wcat.reset_index(inplace = True, drop = True)

# View the first 10 rows after sorting
print("First 10 rows after sorting by waist:")
wcat.head(10)

# Exploratory Data Analysis (EDA) - Visualize outliers with boxplots
print("Boxplots to visualize outliers:")
wcat.plot(kind = 'box', subplots = True, sharey = False, figsize = (8, 6))
plt.subplots_adjust(wspace = 0.75)  # Adjust spacing between subplots
plt.show()

# Split data into target variable (AT) and predictor variable (Waist)
X = pd.DataFrame(wcat['Waist'])  # Create DataFrame for predictor
Y = pd.DataFrame(wcat['AT'])    # Create DataFrame for target variable

# Select numeric features for data preprocessing
numeric_features = ['Waist']

# Define a pipeline for numeric columns:
#  - Impute with mean
#  - Winsorize outliers
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'mean')),
    ('winsor', Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5)),
])

# Wrap it in a ColumnTransformer (in case you have multiple columns later)
preprocessor = ColumnTransformer([('num', num_pipeline, numeric_features)])

# Fit preprocessing pipelines to data
Clean_data = preprocessor.fit(X)

wcat["Waist"] = pd.DataFrame(Clean_data.transform(X))  # Transform waist column with imputation


# Visualize outliers after preprocessing
print("Boxplots to visualize outliers after preprocessing:")
wcat.plot(kind = 'box', subplots = True, sharey = False, figsize = (8, 6))
plt.subplots_adjust(wspace = 0.75) # Adjust spacing between subplots
plt.show()

# Graphical Representation of the Data
# Bar Graph of Target Variable (AT)
plt.figure(figsize = (10, 6)) # Set figure size for better visualization
plt.bar(height = wcat.AT, x = np.arange(1, 110, 1)) # Create bar graph with index as x-axis
plt.xlabel('Index') # Label the x-axis
plt.ylabel('AT Value') # Label the y-axis
plt.title('Bar Graph of AT Values') # Add a title for clarity
plt.show() # Display the bar graph

# Histogram of Target Variable (AT)
plt.figure(figsize = (10, 6)) # Set figure size for better visualization
plt.hist(wcat.AT) # Create a histogram of AT values
plt.xlabel('AT Value') # Label the x-axis
plt.ylabel('Frequency') # Label the y-axis
plt.title('Histogram of AT Values') # Add a title for clarity
plt.show() # Display the histogram

# Bar Graph of Predictor Variable (Waist)
plt.figure(figsize = (10, 6)) # Set figure size for better visualization
plt.bar(height = wcat.Waist, x = np.arange(1, 110, 1)) # Create bar graph with index as x-axis
plt.xlabel('Index') # Label the x-axis
plt.ylabel('Waist Circumference') # Label the y-axis
plt.title('Bar Graph of Waist Circumference') # Add a title for clarity
plt.show() # Display the bar graph

# Histogram of Predictor Variable (Waist)
plt.figure(figsize = (10, 6)) # Set figure size for better visualization
plt.hist(wcat.Waist) # Create a histogram of waist circumference values
plt.xlabel('Waist Circumference') # Label the x-axis
plt.ylabel('Frequency') # Label the y-axis
plt.title('Histogram of Waist Circumference') # Add a title for clarity
plt.show() # Display the histogram

# Bivariate Analysis - Explore the relationship between Waist and AT
# Scatter Plot
plt.scatter(x = wcat['Waist'], y = wcat['AT'])
plt.xlabel('Waist Circumference')  # Label the x-axis
plt.ylabel('AT Value')  # Label the y-axis
plt.title('Scatter Plot of Waist vs AT')  # Add a title for clarity
plt.show()  # Display the scatter plot

# Correlation Coefficient - Measures the strength of the linear relationship
correlation = np.corrcoef(wcat.Waist, wcat.AT)[0, 1]
print("Correlation Coefficient between Waist and AT:", correlation)
# Values closer to 1 or -1 indicate stronger linear relationships.

# Covariance - Measure the direction of the joint variability
covariance = np.cov(wcat.Waist, wcat.AT)[0, 1]
print("Covariance between Waist and AT:", covariance)
# A positive covariance suggests both variables tend to move in the same direction,
# while a negative covariance suggests they move in opposite directions.

# Heatmap to visualize correlations between all variables
dataplot = sns.heatmap(wcat.corr(), annot = True, cmap = "YlGnBu")  # Create heatmap, YlGnBu - Yellow Green Blue
plt.title('Correlation Heatmap')  # Add a title for clarity
plt.show()  # Display the heatmap

# The above are manual approach to perform Exploratory Data Analysis (EDA). 
# The alternate approach is to Automate the EDA process using Python libraries.
# Auto EDA libraries: dtale
d = dtale.show(wcat)
d.open_browser()

# EDA - Business Insights & Statistical Insights

# Linear Regression Modeling
# 1. Model definition (using statsmodels)
model = smf.ols('AT ~ Waist', data = wcat).fit()  # Fit a simple linear regression model

# 2. Model Summary
print("Linear Regression Model Summary:")
print(model.summary())  # Display various statistics about the model

# 3. Predictions
pred1 = model.predict(pd.DataFrame(wcat['Waist']))  # Predict AT values based on waist

# Error Calculation for the Base Model
# 4. Calculate residuals (errors) for each data point
res1 = wcat.AT - pred1  # Actual AT values - Predicted (Fitted) AT values
print("Mean of residuals (should be close to zero for good fit):", np.mean(res1))

# 5. Calculate model evaluation metrics
res_sqr1 = res1 * res1  # Square the residuals
mse1 = np.mean(res_sqr1)  # Mean Squared Error
rmse1 = np.sqrt(mse1)  # Root Mean Squared Error
print("Root Mean Squared Error (RMSE) for the base model:", rmse1)

# 4. Visualization of the regression line
plt.scatter(wcat.Waist, wcat.AT)  # Plot the original data points
plt.plot(wcat.Waist, pred1, "r")  # Plot the regression line in red
plt.xlabel('Waist Circumference')  # Label the x-axis
plt.ylabel('AT Value')  # Label the y-axis
plt.title('Linear Regression Line (AT ~ Waist)')  # Add a title for clarity
plt.legend(['Observed data', 'fitted line'])  # Add a legend
plt.show()  # Display the plot


# Model Tuning with Transformations
# 1. Log Transformation of Predictor Variable
# - Visualize relationship after log transformation
# Scatter Plot with Log-Transformed Waist
plt.scatter(x = np.log(wcat['Waist']), y = wcat['AT'], color = 'brown')  # Create a scatter plot with log-transformed Waist and AT
plt.xlabel('Log(Waist Circumference)')  # Set the x-axis label
plt.ylabel('AT Value')  # Set the y-axis label
plt.title('Scatter Plot with Log-Transformed Waist')  # Set the title of the plot
plt.show()  # Display the plot

# Calculate correlation for transformed data
print("Correlation after log transformation of Waist:", np.corrcoef(np.log(wcat.Waist), wcat.AT)[0, 1])  # Print the correlation coefficient

# Fit Linear Regression with Log-Transformed Predictor
model2 = smf.ols('AT ~ np.log(Waist)', data = wcat).fit()  # Fit a linear regression model with log-transformed Waist
print("Model Summary for log-transformed model:")  # Print model summary message
print(model2.summary())  # Display the model summary

pred2 = model2.predict(pd.DataFrame(wcat['Waist']))  # Predict AT values based on waist

# Error Calculation for Log-Transformed Model
res2 = wcat.AT - pred2  # Calculate residuals
res_sqr2 = res2 * res2  # Square the residuals
mse2 = np.mean(res_sqr2)  # Calculate mean squared error
rmse2 = np.sqrt(mse2)  # Calculate root mean squared error
print("RMSE for log-transformed model:", rmse2)  # Print the RMSE

# Predictions and Visualization for Log-Transformed Model
plt.scatter(np.log(wcat.Waist), wcat.AT)  # Scatter plot of log-transformed Waist vs. AT
plt.plot(np.log(wcat.Waist), pred2, "r")  # Plot the regression line
plt.xlabel('Log(Waist Circumference)')  # Set the x-axis label
plt.ylabel('AT Value')  # Set the y-axis label
plt.title('Regression Line with Log-Transformed Waist')  # Set the title of the plot
plt.legend(['Observed data', 'Fitted line'])  # Add legend to the plot
plt.show()  # Display the plot

# Scatter Plot with Exponential-Transformed AT
plt.scatter(x = wcat['Waist'], y = np.log(wcat['AT']), color = 'orange')  # Create a scatter plot with Waist and log-transformed AT
plt.xlabel('Waist Circumference')  # Set the x-axis label
plt.ylabel('Log(AT Value)')  # Set the y-axis label
plt.title('Scatter Plot with Exponential-Transformed AT')  # Set the title of the plot
plt.show()  # Display the plot

# Calculate correlation for transformed data
print("Correlation after exponential transformation of AT:", np.corrcoef(wcat.Waist, np.log(wcat.AT))[0, 1])  # Print the correlation coefficient

# Fit Linear Regression with Exponential-Transformed Response
model3 = smf.ols('np.log(AT) ~ Waist', data = wcat).fit()  # Fit a linear regression model with Waist and log-transformed AT
print("Model Summary for exponential-transformed model:")  # Print model summary message
print(model3.summary())  # Display the model summary

pred3 = model3.predict(pd.DataFrame(wcat['Waist']))  # Generate predictions based on the model

# Error Calculation for Exponential-Transformed Model
pred3_at = np.exp(pred3)  # Convert predicted log values back to AT values
res3 = wcat.AT - pred3_at  # Calculate residuals
res_sqr3 = res3 * res3  # Square the residuals
mse3 = np.mean(res_sqr3)  # Calculate mean squared error
rmse3 = np.sqrt(mse3)  # Calculate root mean squared error
print("RMSE for exponential-transformed model:", rmse3)  # Print the RMSE

# Predictions and Visualization for Exponential-Transformed Model
plt.scatter(wcat.Waist, np.log(wcat.AT))  # Scatter plot of Waist vs. log-transformed AT
plt.plot(wcat.Waist, pred3, "r")  # Plot the regression line
plt.xlabel('Waist Circumference')  # Set the x-axis label
plt.ylabel('Log(AT Value)')  # Set the y-axis label
plt.title('Regression Line with Exponential-Transformed AT')  # Set the title of the plot
plt.legend(['Observed data', 'Predicted line'])  # Add legend to the plot
plt.show()  # Display the plot

# Comparing Model Performance
print("The base model has a RMSE of:", rmse1)  # Print RMSE of the base model
print("The log-transformed model has a RMSE of:", rmse2)  # Print RMSE of the log-transformed model
print("The exponential-transformed model has a RMSE of:", rmse3)  # Print RMSE of the exponential-transformed model

# Conclusion (based on RMSE comparison, choose the best model)
# Based on the RMSE values, you can choose the model that performs best. 
# A lower RMSE indicates a better fit for the data.

# Note: This code demonstrates trying different transformations. 
# You can explore other transformations or techniques to improve the model's performance.

# Fit a polynomial regression model
model4 = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = wcat).fit()  # Fit the model using log-transformed AT and polynomial terms of Waist
model4.summary()  # Display the summary of the model

# Make predictions using the polynomial model
pred4 = model4.predict(pd.DataFrame(wcat))  # Generate predictions based on the model

# Calculate errors for the polynomial model
pred4_at = np.exp(pred4)  # Transform predictions back to original scale
res4 = wcat.AT - pred4_at  # Calculate residuals
res_sqr4 = res4 * res4  # Square the residuals
mse4 = np.mean(res_sqr4)  # Calculate mean squared error
rmse4 = np.sqrt(mse4)  # Calculate root mean squared error
rmse4  # Display the RMSE

# Visualize the regression lines for the polynomial and linear models
plt.scatter(wcat['Waist'], np.log(wcat['AT']))  # Scatter plot of Waist vs. log-transformed AT
plt.plot(wcat['Waist'], pred4, color = 'red')  # Plot the polynomial regression line
plt.plot(wcat['Waist'], pred3, color = 'green', label = 'linear')  # Plot the linear regression line
plt.legend(['Transformed Data', 'Polynomial Regression Line', 'Linear Regression Line'])  # Add legend to the plot
plt.show()  # Display the plot


# Create a table to compare RMSE of different models
data = {"MODEL": pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE": pd.Series([rmse1, rmse2, rmse3, rmse4])}  # Create a dictionary with model names and RMSE values
table_rmse = pd.DataFrame(data)  # Create a DataFrame to display the RMSE values
table_rmse  # Display the RMSE comparison table

X = wcat[['Waist']]  # Input
y = wcat['AT']       # Target

# ---------- TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 30)

# Visualize the data points in the training and testing sets
plt.scatter(X_train.Waist, np.log(y_train.AT))  # Scatter plot of Waist vs. log-transformed AT in the training set
plt.scatter(X_test.Waist, np.log(y_test.AT))  # Scatter plot of Waist vs. log-transformed AT in the testing set

 # Fit the final model using the training data (ensuring both Waist & AT are in the same DataFrame)
finalmodel = smf.ols(
    'np.log(AT) ~ Waist + I(Waist*Waist)', 
    data = pd.concat([X_train, y_train], axis = 1)
).fit()

# Make predictions on the test data using the final model
test_pred = finalmodel.predict(X_test)  # Generate predictions on the test set
pred_test_AT = np.exp(test_pred)  # Transform predictions back to original scale

# Model Evaluation on Test Data

# Calculate the error (residuals) between actual AT values and predicted AT values on the test set
test_res = y_test.AT - pred_test_AT  # Residuals (errors) for test data
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)  # Average of squared errors
test_rmse = np.sqrt(test_mse)
print("Test RMSE:", test_rmse)  # Print the RMSE on the test set

# Predictions on the Train Data

# Make predictions on the training data using the final model
train_pred = finalmodel.predict(pd.DataFrame(X_train['Waist']))

# Convert the predicted log values back to AT scale for the training data
pred_train_AT = np.exp(train_pred)

# Model Evaluation on Train Data 

# Calculate the error (residuals) between actual AT values and predicted AT values on the training set
train_res = y_train.AT - pred_train_AT  # Residuals (errors) for training data
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)  # Average of squared errors
train_rmse = np.sqrt(train_mse)
print("Train RMSE:", train_rmse)  # Print the RMSE on the training set


# Create a pipeline that:
#  1) applies the preprocessor
#  2) then adds polynomial features
#  3) then fits a linear regression
pipe = Pipeline([
    ('preprocess', preprocessor),
    ('poly', PolynomialFeatures()),
    ('linreg', LinearRegression())
])

# Define hyperparams to tune:
#  - 'poly__degree': polynomial degrees to try
#  - you could also add more hyperparams from LinearRegression or other models
param_grid = {
    'poly__degree': [1, 2, 3, 4]  # test polynomials of degree 1-4
}

# ---------- GRID SEARCH CV ----------
grid_search = GridSearchCV(pipe, param_grid, cv = 5, scoring = 'neg_root_mean_squared_error') # scoring 'neg_mean_squared_error', 'r2'
'''
Why use 'neg_root_mean_squared_error' instead of 'root_mean_squared_error'?
GridSearchCV assumes that higher scores are better.

But metrics like RMSE (Root Mean Squared Error) are loss functions — lower is better.

So scikit-learn negates them (i.e., multiplies them by -1) so that lower RMSE becomes a higher score (less negative).

That’s why we use 'neg_root_mean_squared_error'.
'''

grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best Score (RMSE * -1):", -1 * grid_search.best_score_) # Scikit-learn's GridSearchCV automatically negates loss metrics like RMSE or MSE to fit into its "higher is better" logic.

# Evaluate on test set:
from sklearn.metrics import root_mean_squared_error
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = root_mean_squared_error(y_test, y_pred)
print("Test RMSE:", rmse_test)

# Combine X_train + X_test for the final full dataset
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

# Use the best hyperparams from the grid search
final_degree = grid_search.best_params_['poly__degree']

# Build a final pipeline with the best degree
final_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('poly', PolynomialFeatures(degree = final_degree)), 
    ('linreg', LinearRegression())
])

# Fit on the entire dataset
final_pipe.fit(X_full, y_full)

# Save the final pipeline
import joblib
joblib.dump(final_pipe, 'final_polynomial_pipeline.pkl')
print("Saved the final pipeline to final_polynomial_pipeline.pkl")

import os
os.getcwd()

# END of Simple Linear Regression. Now get into deployment. 