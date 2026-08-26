\
'''CRISP-ML(Q): 6 phases 

1.a. Business Understanding
Business Problem:
    There are a lot of assumptions in the diagnosis pertaining to cancer. In a few cases radiologists, 
    pathologists and oncologists go wrong in diagnosing whether tumor is benign (non-cancerous) or malignant (cancerous). 

High Level Solution: 
    Hence team of physicians want us to build an AI application which will predict with confidence the presence of cancer 
    in a patient. This will serve as a compliment to the physicians.

Business Objective(s): Maximize Cancer Detection
Business Constraint(s): Minimize Treatment Cost & Maximize Patient Convenience

Success Criteria: 
    Must be quantitative and numbers must be arrived at based on previous experience and in consultation 
    with project sponsor.
    Business success criteria: Increase the correct diagnosis of cancer in at least 96% of patients
    Machine Learning success criteria: Achieve an accuracy of atleast 98%
    Economic success criteria: Reducing medical expenses will improve trust of patients and thereby hospital will see 
    an increase in revenue by atleast 12%

* Perform research on similar project either within your company or resort to external research forums (e.g., Google Scholar) 

* HLD - DAR - DLD (Watch "AI for Business Professionals" module on AiTutor LMS to gain more knowledge).

* Create Project Charter, which contains details at a high level. 

1.b. Data Understanding: 
    
Data Collection: Data Sources -> Data Storage (DB, DWH, DL, DLH) -> EDA (Business Insights & Statistical Insights)

Meta Data Description: 
    Data is collected from the hospital for 569 patients. 30 features and 1 label comprise the feature set. 
    Ten real-valued features are computed for each cell nucleus:

    a) radius (mean of distances from center to points on the perimeter)
    b) texture (standard deviation of gray-scale values)
    c) perimeter
    d) area
    e) smoothness (local variation in radius lengths)
    f) compactness (perimeter^2 / area - 1.0)
    g) concavity (severity of concave portions of the contour)
    h) concave points (number of concave portions of the contour)
    i) symmetry
    j) fractal dimension ("coastline approximation" - 1)
    k) Diagnosis (Label/Target/Output) - 2 levels (B = Benign; M = Malignant)'''

# CODE MODULARITY IS EXTREMELY IMPORTANT

# Import the libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For data visualization

# Import necessary modules from scikit-learn
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler  # For encoding and scaling
from feature_engine.outliers import Winsorizer  # For outlier treatment
from sklearn.compose import ColumnTransformer  # For column-wise transformations
from sklearn.model_selection import train_test_split, GridSearchCV  # For data splitting and hyperparameter tuning
from sklearn.neighbors import KNeighborsClassifier  # For kNN model
from sklearn.pipeline import Pipeline  # For chaining transformations and model
import sklearn.metrics as skmet  # For evaluation metrics
import pickle  # For saving the model
from imblearn.over_sampling import SMOTE  # For balancing data
import dtale  # For EDA GUI
from sqlalchemy import create_engine  # For SQL connection
from urllib.parse import quote  # To encode password
import joblib  # For saving pipeline
import shap  # For explainability
from sklearn.preprocessing import LabelEncoder  # For label encoding


# Reading the cancer data from a CSV file
cancerdata = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\Machine_learning\supervised_machine_learning\KNN_Complete_Pipeline\cancerdata.csv")

# MySQL connection setup
user = '****'
pw = quote('**********')l
db = 'cancer_db'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
cancerdata.to_sql('cancer', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Load data from MySQL
sql = 'select * from cancer'
cancerdf = pd.read_sql_query(sql, con = engine)
print(cancerdf)

# Launch D-Tale GUI for EDA - AutoEDA
d = dtale.show(cancerdf)
d.open_browser()

# Create two key documents - Business Insights & Statistical Insights

# Phase 2 of CRISP-ML(Q) "Data Preparation"

# Recode diagnosis labels
cancerdf['diagnosis'] = np.where(cancerdf['diagnosis'] == 'B', 'Benign', cancerdf['diagnosis'])
cancerdf['diagnosis'] = np.where(cancerdf['diagnosis'] == 'M', 'Malignant', cancerdf['diagnosis'])

# Drop ID column
cancerdf.drop(['id'], axis = 1, inplace = True)

# Dataset overview
cancerdf.info()
cancerdf.describe()

# Separate features and target
cancerdf_X = pd.DataFrame(cancerdf.iloc[:, 1:])
cancerdf_y = pd.DataFrame(cancerdf.iloc[:, 0])
cancerdf_X.info()

# Identify numerical and categorical features
numeric_features = cancerdf_X.select_dtypes(exclude = ['object']).columns
categorical_features = cancerdf_X.select_dtypes(include = ['object']).columns

# Create preprocessing pipelines
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'mean')),
    ('winsorize', Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5)),
    ('scale', MinMaxScaler())
])

categ_pipeline = Pipeline([
    ('encoding', OneHotEncoder(drop = 'first'))
])

# Combine pipelines into a column transformer
preprocess_pipeline = ColumnTransformer([
    ('categorical', categ_pipeline, categorical_features),
    ('numerical', num_pipeline, numeric_features)
])

# ---------------- FEATURE SELECTION ------------------
# Watch "Hypothesis testing" self paced module from LMS

# Importing SelectKBest for feature selection
from sklearn.feature_selection import SelectKBest, f_classif 
# f_classif – a scoring function used specifically for classification tasks
# scoring function (like f_classif, chi2, or mutual_info_classif) - 
# f_classif is the ANOVA F-value between:each feature and the target variable (y)
# Higher F-value = more discriminatory power = better feature.

# Creating the feature selector separately so we can access it later
selector = SelectKBest(score_func = f_classif, k = 10)

# Creating full pipeline: preprocessing + feature selection
full_pipeline = Pipeline([
    ('preprocess', preprocess_pipeline),
    ('feature_selection', selector)
])


# Fit pipeline and transform features
X_selected = full_pipeline.fit_transform(cancerdf_X, cancerdf_y.values.ravel()) # .ravel() is a NumPy function that flattens a multi-dimensional array into a 1D array.
X_selected

##############################################################
# Ravel function explanation
import numpy as np
y = np.array([[1], [0], [1], [1]])  # shape = (4, 1)
print("Original shape:", y.shape)

y_flat = y.ravel()
print("Flattened shape:", y_flat.shape)
print("Flattened array:", y_flat)
##############################################################

# Save pipeline for deployment
joblib.dump(full_pipeline, 'pipeline_with_feature_selection')

# Extract feature names after transformation
preprocess_pipeline.fit(cancerdf_X) # Fit only the preprocessing pipeline

preprocessor = full_pipeline.named_steps['preprocess'] 
# named_steps is a dictionary that lets you access individual steps by their names: preprocess, feature_selection
preprocessor

categorical_transformer = preprocessor.transformers_[0][1] 
# Get the transformer for categorical features (assumed to be first in the list)
categorical_transformer

'''
preprocess_pipeline = ColumnTransformer([
    ('categorical', categ_pipeline, categorical_features),
    ('numerical', num_pipeline, numeric_features)
])
'''

encoded_cat_cols = categorical_transformer.named_steps['encoding'].get_feature_names_out(categorical_features)
encoded_cat_cols # Get encoded column names

numeric_cols = numeric_features # Get numeric columns (unchanged)
numeric_cols

all_transformed_cols = np.concatenate([encoded_cat_cols, numeric_cols])
all_transformed_cols # Combined column names from both transformations

# Get mask of selected features
selected_mask = selector.get_support() 
# This line returns a boolean mask (array of True/False values) indicating which features were 
# selected by SelectKBest.
selected_mask

selected_feature_names = all_transformed_cols[selected_mask]
selected_feature_names

cancerclean = pd.DataFrame(X_selected, columns = selected_feature_names) # Create DataFrame with selected columns
cancerclean

# Show info about cleaned data
cancerclean.info()
res = cancerclean.describe()

# Prepare target variable
Y = np.array(cancerdf_y['diagnosis'])
Y_1 = cancerdf_y['diagnosis']
Y_1.value_counts()

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(cancerclean, Y, test_size = 0.2, random_state = 24)

# SMOTE for balancing - Synthetic Minority Over-sampling Technique
# Borderline-SMOTE; SMOTE-NC (SMOTE for Nominal and Continuous); ADASYN (Adaptive Synthetic Sampling); KMeans-SMOTE; Safe-Level-SMOTE; SVMSMOTE
'''
For each minority sample, SMOTE:

Finds its k nearest neighbors from within the minority class (default k = 5).

Randomly picks one of those neighbors.

Draws a straight line between the original sample and the chosen neighbor.

Then it randomly picks a point along that line to create a new, synthetic sample.'''

smote = SMOTE(random_state = 42) # You’re creating a SMOTE object from the imblearn library

X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)
Y_resampled_1 = pd.Series(Y_resampled)
Y_resampled_1.value_counts()
X_train = X_resampled
Y_train = Y_resampled
# data should be balanced before odel bulding 

# Phase 3 of CRISP-ML(Q) - Model Building. 
# Train kNN classifier
knn = KNeighborsClassifier(n_neighbors = 21) # thumb rule is sqrt(n/2) to decide the 'k' value
KNN = knn.fit(X_train, Y_train)
pred_train = knn.predict(X_train)
print(skmet.accuracy_score(Y_train, pred_train))

pred_test = knn.predict(X_test)
print(skmet.accuracy_score(Y_test, pred_test))
pd.crosstab(Y_test, pred_test, rownames = ['Actual'], colnames = ['Predictions'])

cm = skmet.confusion_matrix(Y_test, pred_test)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title = 'Cancer Detection - Confusion Matrix', xlabel = 'Predicted Value', ylabel = 'Actual Value')

# Test kNN performance for different k values
acc = []
for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
    
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")

# Hyperparameter tuning using GridSearchCV
k_range = list(range(3, 50, 2))
param_grid = dict(n_neighbors = k_range)
grid = GridSearchCV(knn, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 2)
KNN_new = grid.fit(X_train, Y_train)
print(KNN_new.best_params_)
accuracy = KNN_new.best_score_ * 100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))
pred_test = KNN_new.predict(X_test)
cm = skmet.confusion_matrix(Y_test, pred_test)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title = 'Cancer Detection - Confusion Matrix', xlabel = 'Predicted Value', ylabel = 'Actual Value')

# Save best estimator
knn_best = KNN_new.best_estimator_
knn_best
pickle.dump(knn_best, open('knn.pkl', 'wb'))

import os
os.getcwd()

# Explainability is based on data features and output; Interpretability is based on model parameters
# SHAP explainability SHapley Additive exPlanations

# pip install shap
import shap

# Label encoding for target (if needed)
label_encoder = LabelEncoder() 

'''
Converts categorical target labels (`"Benign"`, `"Malignant"`) into numeric labels (`0`, `1`)
This is not mandatory for SHAP itself (since SHAP uses model output), 
but it’s good practice for plotting or analyzing class distributions.
'''

Y_train_num = label_encoder.fit_transform(Y_train)
Y_test_num = label_encoder.transform(Y_test)

# Create sample_X as DataFrame with proper column names
sample_X = pd.DataFrame(X_test, columns = cancerclean.columns)

# Wrap SHAP inputs into DataFrame inside predict function
def knn_predict_proba(X):
    if isinstance(X, np.ndarray): # Checks if SHAP passed a NumPy array
        X = pd.DataFrame(X, columns = sample_X.columns) # Converts array to DataFrame with correct column names
    return knn_best.predict_proba(X)[:, 1] # Gets class probabilities from your model. Each row represents [prob_class_0, prob_class_1]

'''
Above function is explained here
SHAP may pass a NumPy array (ndarray) as input, but,
the KNN model (knn_best) was trained on a DataFrame with named columns (like 'radius_mean', 'texture_mean', etc.),
if you pass a raw NumPy array without column names, scikit-learn may throw this warning:
UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names,
So this line re-wraps X into a pandas DataFrame using the correct column names (from sample_X.columns) only if needed.
'''

# Initialize KernelExplainer
explainer = shap.KernelExplainer(knn_predict_proba, sample_X) # knn_predict_proba: Your custom prediction function (returns class 1 probabilities)

'''
Creates a SHAP explainer object using the KernelExplainer, which is:
Model-agnostic (works with any model — like KNN, SVM, logistic regression, etc.)
Based on LIME-like principles using Shapley values
SHAP will take sample_X as a reference and vary each feature individually to see how much it contributes to the predicted outcome.
'''

# Compute SHAP values
shap_values = explainer.shap_values(sample_X)
shap_values

# Sample demonstration: Computes SHAP values for each row (sample)

'''
For each prediction:
It explains how much each feature pushed the prediction higher or lower compared to the 
average prediction (baseline)
E.g. If you have 3 samples (rows) and 4 features (columns):
    
shap_values =
[[ 0.2,  0.1, -0.05,  0.0  ],
 [-0.1, 0.25,  0.05, -0.2  ],
 [ 0.3, -0.1, 0.05,   0.0  ]]

For the 1st prediction, Feature 1 pushed the model output up by 0.2, Feature 3 down by 0.05, etc.
'''

# Plot summary
shap.summary_plot(shap_values, sample_X)

'''
Visualizes the SHAP values using a summary plot

Shows:
    The most important features overall (ranked top-down)
    The impact of each feature (positive/negative SHAP values)
    The distribution of feature effects (color shows actual feature values)
    Horizontal bars sorted top to bottom (most important features on top)
    Each dot = one data point's SHAP value for that feature
    Color = whether the actual value of that feature was high (red) or low (blue)
'''

# ---------------- END OF SHAP ------------------

# Toy Example for SHAP understanding

''' Predict house prices based on just 2 features: Bedrooms, Size (sqft)'''

Bedrooms = 2
Size = 1000

Price = 50000 + (30000 * Bedrooms) + (200 * Size) # Equation obtained by running Linear Regression 
Price

# Step 1: Baseline Prediction (no features known)
# Let’s say the average predicted price across all houses is:
Base_value = 200000

# Step 2: Add one feature at a time, calculate contribution
# Add 'Bedroom' and ignore 'Size'
Price = 50000 + (30000 * Bedrooms)
Price
110000 - 50000 # Change from base value is 60000. So contribution from Bedrooms = 60000

# Add Size and ignore Bedroom
Price = 50000 + (200 * Size)
Price
250000 - 50000 # Change from base value is 200000. So contribution from Size = 200000

'''
Shapley values average these contributions in all possible orders:
First add Bedrooms → then Size

First add Size → then Bedrooms

Order	              Contribution from Bedrooms	          Contribution from Size
Add Bedrooms first	         60000	                                200000
Add Size first	             60000	                                200000

Shapley Value = Average of both orders
Bedrooms = 60000 (Average SHAP Value)
Size = 200000 (Average SHAP Value)
'''

# END OF KNN CODE