
'''CRISP-ML(Q): 6 phases 

1.a. Business Understanding
Business Problem:
    Insurance companies face challenges in predicting whether a claimant will hire an attorney. 
    This impacts settlement costs, claim processing time, and resource allocation. 
    Attorney involvement typically increases claim costs and processing complexity.

High Level Solution: 
    Build an AI application to predict the likelihood of attorney involvement in insurance claims.
    This will help the insurance company proactively manage claims, allocate resources efficiently,
    and prepare appropriate settlement strategies.

Business Objective(s): Maximize early detection of attorney involvement
Business Constraint(s): Minimize false negatives (missing attorney cases) & Optimize claim processing costs

Success Criteria: 
    Must be quantitative and numbers must be arrived at based on previous experience and in consultation 
    with project sponsor.
    Business success criteria: Reduce claim processing time by 20% through early attorney detection
    Machine Learning success criteria: Achieve an accuracy of at least 85% with high recall for attorney cases
    Economic success criteria: Reduce settlement costs by 15% through better claim management
    
* Perform research on similar project either within your company or resort to external research forums (e.g., Google Scholar) 

* HLD - DAR - DLD (Watch "AI for Business Professionals" module on AiTutor LMS to gain more knowledge).

* Create Project Charter, which contains details at a high level. 

1.b. Data Understanding: 
    
Data Collection: Data Sources -> Data Storage (DB, DWH, DL, DLH) -> EDA (Business Insights & Statistical Insights)

Meta Data Description: 
    Data is collected from insurance claims for 1341 cases. The dataset contains the following features:
    
    a) CASENUM - Unique case identifier
    b) ATTORNEY - Target variable (0 = No attorney, 1 = Attorney hired)
    c) CLMSEX - Claimant's gender (0 = Male, 1 = Female)
    d) CLMINSUR - Claimant's insurance status (0 = Uninsured, 1 = Insured)
    e) SEATBELT - Seatbelt usage (0 = No, 1 = Yes)
    f) CLMAGE - Claimant's age in years
    g) LOSS - Financial loss amount in thousands of dollars'''

# CODE MODULARITY IS EXTREMELY IMPORTANT

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sklearn.metrics as skmet
import pickle
import joblib

# Feature selection
from sklearn.feature_selection import f_classif, SelectKBest  # ANOVA F-test for feature importance

# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score

# Explainability
import shap

# AutoEDA
import dtale
# SQL
from sqlalchemy import create_engine
from urllib.parse import quote

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHASE 1.b: DATA COLLECTION & UNDERSTANDING
# =============================================================================

# Reading the claimants data from a CSV file
claimants_data = pd.read_csv(r"C:\MachineLearningModels\claimants.csv")  # Load raw CSV data

# MySQL connection setup
user = '******'  # Database username
pw = quote('*************')  # URL encode password for special characters
db = 'db1'  # Database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")  # Create database connection engine

# Write data to MySQL database
claimants_data.to_sql('claimants', con=engine, if_exists='replace', chunksize=1000, index=False)  # Save data to DB in chunks

# Load data from MySQL and convert to appropriate data types
sql = 'select * from claimants'  # SQL query to retrieve all data
claimants_df = pd.read_sql_query(sql, con=engine).convert_dtypes()  # Read from DB and auto-convert dtypes
print(claimants_df.head())  # Display first 5 rows

# Launch D-Tale GUI for interactive EDA
d = dtale.show(claimants_df)  # Initialize D-Tale with dataframe
d.open_browser()  # Open interactive EDA interface in browser

# =============================================================================
# AUTOEDA INSIGHTS (Key findings from D-Tale exploration)
# =============================================================================

business_insights = """
1. Class Distribution: Attorney vs. Non‑attorney is roughly balanced (~49% vs ~51%), so there is no major class imbalance risk.
2. Age Distribution: Claimant ages range from 0 to 95, with a wide spread; different age groups may show different attorney‑hiring behavior.
3. Loss Amount: LOSS ranges from 0 to ~173.6 with very high variance, indicating a few very high‑cost claims that can strongly influence business risk.
4. Seatbelt Usage: SEATBELT has a very low mean (~1.7% “Yes”), suggesting most claimants did not report wearing seatbelts; this may impact claim legitimacy and negotiations.
5. Insurance Status: CLMINSUR has a high mean (~0.91), indicating most claimants are insured, which influences how claims are managed and settled.
6. Gender Patterns: CLMSEX is slightly skewed (~56% coded as 1), so any gender effects on attorney involvement are likely modest.
7. Data Quality: Missing data is concentrated in CLMAGE and to a lesser extent CLMINSUR, SEATBELT, and CLMSEX, so imputation is required before modeling.
8. Outliers: LOSS shows extreme values relative to its median, so outlier treatment (e.g., winsorization) is recommended for stable and robust modeling.
"""
print(business_insights)

189/1340*100

# =============================================================================
# STATISTICAL INSIGHTS FOR MODEL BUILDING
# =============================================================================

# Basic statistics
print("\nDataset Shape:", claimants_df.shape)
print("\nData Types:")
print(claimants_df.dtypes)

print("\nMissing Values:")
print(claimants_df.isnull().sum())
print(f"\nMissing Percentage:\n{(claimants_df.isnull().sum() / len(claimants_df) * 100).round(2)}")

print("\nDescriptive Statistics:")
description=claimants_df.describe()
print(description=claimants_df.describe())

print("\nTarget Variable Distribution:")
print(claimants_df['ATTORNEY'].value_counts())
print(f"\nClass Balance Ratio:\n{claimants_df['ATTORNEY'].value_counts(normalize=True)}")

# Correlation analysis
print("\nCorrelation with Target (ATTORNEY):")
correlation_with_target = claimants_df.corr()['ATTORNEY'].sort_values(ascending=False)
print(correlation_with_target)

statistical_insights = """
STATISTICAL INSIGHTS:
1. Dataset contains 1340 observations with 7 features
2. Missing values present in CLMAGE (~14.1%), SEATBELT (~3.6%), CLMINSUR (~3.1%), CLMSEX (~0.9%)
3. Target variable (ATTORNEY) is WELL-BALANCED (51.1% vs 48.9%) - NO balancing needed
4. LOSS shows high variance and potential outliers - winsorization recommended
5. Binary categorical features (CLMSEX, CLMINSUR, SEATBELT) preserved as integers
6. Age ranges from 0 to 95 years with 14.1% missing values - imputation required
7. LOSS amount varies significantly (0 to 173.6) - scaling essential
8. LOSS has weak negative correlation (-0.22) with ATTORNEY
"""
print(statistical_insights)

# =============================================================================
# PHASE 2: DATA PREPARATION
# =============================================================================

# Drop ID column (not useful for prediction)
claimants_df.drop(['CASENUM'], axis=1, inplace=True)  # Remove CASENUM column in-place

# Display dataset structure and statistics
claimants_df.info()  # Show data types, non-null counts, memory usage
claimants_df.describe()  # Display statistical summary of numerical columns

# Separate features (X) and target variable (y)
X = claimants_df.drop(['ATTORNEY'], axis=1)  # All columns except target
y = claimants_df['ATTORNEY']  # Target variable only

print("\nFeatures shape:", X.shape)  # Print number of rows and features
print("Target shape:", y.shape)  # Print number of target values

# Identify categorical and numerical features
# Categorical features (binary encoded as 0/1)
categorical_features = ['CLMSEX', 'CLMINSUR', 'SEATBELT']  # Binary categories: gender, insurance, seatbelt

# Numerical features (continuous values)
numeric_features = ['CLMAGE', 'LOSS']  # Continuous: age and loss amount

print("\nCategorical features (binary):", categorical_features)  # Display categorical feature list
print("Numerical features (continuous):", numeric_features)  # Display numerical feature list

# =============================================================================
# CREATE PREPROCESSING PIPELINE
# =============================================================================

# Categorical pipeline - mode imputation only (no scaling for binary features)
categ_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent'))  # Fill missing values with mode
])

# Numerical pipeline - median imputation, outlier treatment, and standardization
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),  # Fill missing values with median (robust to outliers)
    ('winsorize', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),  # Cap outliers at IQR boundaries
    ('scale', StandardScaler())  # Standardize to mean=0, std=1 (required for logistic regression)
])

# Combine pipelines for different feature types
preprocess_pipeline = ColumnTransformer([
    ('categorical', categ_pipeline, categorical_features),  # Apply categorical pipeline to binary features
    ('numerical', num_pipeline, numeric_features)  # Apply numerical pipeline to continuous features
])

# =============================================================================
# FEATURE SELECTION ANALYSIS - ANOVA F-TEST
# =============================================================================

# Apply preprocessing to features
X_preprocessed = preprocess_pipeline.fit_transform(X, y)  # Fit pipeline on data and transform

# Create feature names list and convert to DataFrame
all_features = categorical_features + numeric_features  # Combine feature lists in order
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=all_features)  # Create DataFrame with column names

# Calculate ANOVA F-statistics for feature importance
f_scores, p_values = f_classif(X_preprocessed_df, y)  # Calculate F-statistic and p-values
f_df = pd.DataFrame({
    'Feature': all_features,  # Feature names
    'F_Score': f_scores,  # F-statistic scores
    'P_Value': p_values  # Statistical significance p-values
}).sort_values('F_Score', ascending=False)  # Sort by F-score (highest first)

print("\nANOVA F-Test Scores:")  # Print header
print(f_df)  # Display feature importance table

""" 
1.Based on the ANOVA F-test, LOSS is the most important feature and has a very strong, statistically significant relationship with attorney involvement.
2.CLMINSUR, CLMSEX, and SEATBELT also show statistically significant differences between attorney vs non-attorney groups, so they are useful predictors.
3.CLMAGE does not show a statistically significant difference between the classes (high p-value), so its direct effect seems weak in this dataset
"""

# Save feature importance scores for reference
f_df.to_csv('feature_importance_scores.csv', index=False)  # Save to CSV file
print("✓ Feature importance scores saved: feature_importance_scores.csv")  # Confirmation message

# Create final preprocessing pipeline
final_pipeline = Pipeline([
    ('preprocess', preprocess_pipeline)  # Wrap preprocessing in a pipeline
])

# Transform the data using the pipeline
X_clean = final_pipeline.fit_transform(X, y)  # Fit on all data and transform
X_clean_df = pd.DataFrame(X_clean, columns=all_features)  # Convert to DataFrame with feature names

# Save preprocessing pipeline for deployment
joblib.dump(final_pipeline, 'preprocessing_pipeline_logistic.pkl')  # Serialize pipeline to disk

# =============================================================================
# WRITE CLEAN DATA BACK TO DATABASE
# =============================================================================

print("WRITING CLEAN DATA TO DATABASE")
# Combine clean features with target
clean_data_with_target = X_clean_df.copy()
clean_data_with_target['ATTORNEY'] = y.values

# Save to database table
clean_data_with_target.to_sql('claimants_clean', con=engine, if_exists='replace',  # Write to 'claimants_clean' table
                              chunksize=1000, index=False)  # Process in chunks, exclude index
print("Clean data successfully written to 'claimants_clean' table")  # Confirmation message

# Verify data was written correctly
sql_verify = 'select * from claimants_clean limit 5'  # SQL to get first 5 rows
verify_df = pd.read_sql_query(sql_verify, con=engine)  # Execute query
print("\nSample of clean data from database:")  # Header
print(verify_df)  # Display sample rows

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

# Split data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_clean_df, y,  # Features and target
    test_size=0.2,  # 20% for testing, 80% for training
    random_state=42,  # Fixed seed for reproducibility
    stratify=y  # Maintain same class proportion in train and test
)

print(f"\nTraining set size: {X_train.shape}")  # Display train set dimensions
print(f"Test set size: {X_test.shape}")  # Display test set dimensions
print(f"\nTrain target distribution:\n{y_train.value_counts()}")  # Show class counts in train
print(f"\nTest target distribution:\n{y_test.value_counts()}")  # Show class counts in test

# =============================================================================
# CLASS BALANCING ANALYSIS
# =============================================================================

print(f"Original class distribution:\n{y_train.value_counts()}")  # Show class counts
print(f"\nClass proportions:\n{y_train.value_counts(normalize=True)}")  # Show as percentages

# Calculate imbalance ratio (closer to 1.0 = better balance)
class_counts = y_train.value_counts()  # Get counts for each class
imbalance_ratio = class_counts.max() / class_counts.min()  # Ratio of majority to minority
print(f"\nImbalance ratio: {imbalance_ratio:.2f}")  # Display ratio (1.0 = perfect balance)

balancing_decision = f"""
BALANCING TECHNIQUE DECISION:
- Class 0: {class_counts[0]} samples ({class_counts[0]/len(y_train)*100:.1f}%)
- Class 1: {class_counts[1]} samples ({class_counts[1]/len(y_train)*100:.1f}%)
- Imbalance ratio: {imbalance_ratio:.2f} (1.0 = perfect balance)

ANALYSIS:
The dataset is already well-balanced (ratio >1).
Balancing techniques like SMOTE/ADASYN are NOT needed as they may:
  - Introduce synthetic noise
  - Reduce model generalization
  - Waste computational resources

DECISION: Proceed with original data - NO BALANCING REQUIRED ✓
We'll use class_weight='balanced' in the model for minor adjustments if needed.
"""
print(balancing_decision)


# =============================================================================
# PHASE 3: MODEL BUILDING & HYPERPARAMETER TUNING
# =============================================================================

print("MODEL BUILDING & HYPERPARAMETER TUNING")  # Section header

# Define cross-validation strategy (5-fold stratified)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5 folds, shuffled, maintains class proportion

# Hyperparameter grid for tuning
print("\nStarting hyperparameter tuning...")  # Status message

# Hyperparameter grid for tuning
# We only tune parameters that control regularization behaviour and the solver:
# - C: inverse of regularization strength. Smaller C = stronger regularization (heavier penalty on large coefficients).
#      We search from very strong (0.001) to very weak (100) to let the model find the right complexity.
# - penalty: type of regularization.
#      'l1' (Lasso) can drive some coefficients exactly to zero (feature selection effect).
#      'l2' (Ridge) keeps all features but shrinks coefficients towards zero for stability.
# - solver: optimization algorithm that must be compatible with the chosen penalty.
#      'liblinear' is reliable for small, binary problems.
#      'saga' scales better and also supports both L1 and L2.
# - max_iter: kept fixed at 1000 to ensure convergence for all combinations; not a true hyperparameter to optimize.
print("\nStarting hyperparameter tuning...")  # Status message

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],      # Try different levels of regularization strength
    'penalty': ['l1', 'l2'],                   # Compare L1 vs L2 regularization
    'solver': ['liblinear', 'saga'],           # Use solvers compatible with both L1 and L2
    'max_iter': [1000]                         # Fixed high iteration cap for stable convergence
}

from sklearn.model_selection import GridSearchCV  # Import grid search

# Initialize GridSearchCV with logistic regression
grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42, class_weight='balanced'),  # Base model with balanced class weights
    param_grid=param_grid,  # Hyperparameter combinations to test
    cv=cv_strategy,  # Cross-validation strategy
    scoring='accuracy',  # Metric to optimize
    n_jobs=-1,  # Use all CPU cores for parallel processing
    verbose=1  # Print progress
)

# Fit model and find best hyperparameters
grid_search.fit(X_train, y_train)  # Train on all parameter combinations
results=pd.DataFrame(grid_search.cv_results_)
log_reg = grid_search.best_estimator_  # Extract best performing model

print(f"\nBest Parameters: {grid_search.best_params_}")  # Display optimal hyperparameters
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")  # Display best cross-validation score

# =============================================================================
# PHASE 4: MODEL EVALUATION
# =============================================================================

print("MODEL EVALUATION")  # Section header

# Generate predictions on train and test sets
y_train_pred = log_reg.predict(X_train)  # Predict class labels for training data
y_test_pred = log_reg.predict(X_test)  # Predict class labels for test data
y_test_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # Get probability for positive class (Attorney=1)

# Calculate performance metrics
test_accuracy = skmet.accuracy_score(y_test, y_test_pred)  # Proportion of correct predictions
test_auc = roc_auc_score(y_test, y_test_pred_proba)  # Area under ROC curve (0.5 = random, 1.0 = perfect)
test_f1 = f1_score(y_test, y_test_pred)  # Harmonic mean of precision and recall


print("MODEL PERFORMANCE METRICS")  # Section header
print(f"Test Accuracy: {test_accuracy:.4f}")  # Display accuracy
print(f"Test AUC: {test_auc:.4f}")  # Display AUC score
print(f"Test F1-Score: {test_f1:.4f}")  # Display F1 score

# Display classification report with precision, recall, F1
print(f"\nClassification Report:\n")  # Print header
print(classification_report(y_test, y_test_pred, target_names=['No Attorney', 'Attorney']))  # Show detailed metrics

# Create and visualize confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)  # Calculate confusion matrix
plt.figure(figsize=(8, 6))  # Create figure with specified size
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm_test,  # Create display object
                                     display_labels=['No Attorney', 'Attorney'])  # Set class labels
cmplot.plot(cmap='Blues')  # Plot with blue color scheme
cmplot.ax_.set(title='Attorney Prediction - Confusion Matrix',  # Set plot title
               xlabel='Predicted Value',  # Set x-axis label
               ylabel='Actual Value')  # Set y-axis label
plt.tight_layout()  # Adjust spacing to prevent label cutoff
plt.savefig('confusion_matrix_logistic.png')  # Save figure to file
plt.show()  # Display plot

# Plot ROC curve showing model discrimination ability
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)  # Calculate ROC curve points
plt.figure(figsize=(8, 6))  # Create new figure
plt.plot(fpr, tpr, label=f'AUC = {test_auc:.4f}', linewidth=2)  # Plot ROC curve with AUC in label
plt.plot([0, 1], [0, 1], 'k--', label='Random')  # Diagonal reference line (random classifier)
plt.xlabel('False Positive Rate-how many non-attorney cases are wrongly predicted as attorney')  # Set x-axis label
plt.ylabel('True Positive Rate- how many attorney cases are correctly predicted')  # Set y-axis label
plt.title('ROC Curve')  # Set plot title
plt.legend()  # Show legend
plt.grid(alpha=0.3)  # Add semi-transparent grid
plt.tight_layout()  # Adjust spacing
plt.savefig('roc_curve_logistic.png')  # Save figure to file
plt.show()  # Display plot

# Save trained model to disk
pickle.dump(log_reg, open('logistic_regression_model.pkl', 'wb'))  # Serialize model to binary file
print("\n✓ Model saved: logistic_regression_model.pkl")  # Confirmation message

# =============================================================================
# PHASE 5: MODEL EXPLAINABILITY - SHAP
# =============================================================================

print("MODEL EXPLAINABILITY - SHAP")  # Print section title

# Initialize SHAP explainer for linear models
explainer_shap = shap.LinearExplainer(log_reg, X_train)  # Create explainer using training data
shap_values = explainer_shap.shap_values(X_test)  # Calculate SHAP values for test set

print("✓ SHAP values computed")  # Confirmation message
# Create summary plot showing feature importance
plt.figure(figsize=(10, 6))  # Create figure with specified size
shap.summary_plot(shap_values, X_test, feature_names=all_features, show=False)  # Generate SHAP summary plot
plt.tight_layout()  # Adjust spacing
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')  # Save with high resolution
plt.show()  # Display plot
print("✓ SHAP summary plot saved: shap_summary_plot.png")  # Confirmation message

print("CRISP-ML(Q) PIPELINE COMPLETED")  # Print completion message

"""
Each dot is one person; dots to the right increase predicted attorney risk, dots to the left decrease it.
For LOSS, high values (pink) mostly push predictions strongly positive, confirming LOSS is the dominant driver of attorney involvement.
CLMINSUR, CLMSEX, and SEATBELT have smaller but visible effects, while CLMAGE has relatively low impact, meaning age matters less for the model.

"""