# Machine Learning

A collection of machine learning projects covering supervised and unsupervised learning algorithms, built from data preprocessing through to model deployment with Streamlit and Flask.

Each sub-folder is a self-contained mini-project: it typically includes the training script, the raw/sample dataset, a saved model (`.pkl`) and preprocessing pipeline, and a small web app (Streamlit or Flask) for serving predictions.

## Repository Structure

```
Machine-Learning-
├── supervised_machine_learning/
│   ├── Simple Linear Regression/
│   ├── Multiple Linear Regression/
│   ├── Lasso, Ridge, Elastic Net/
│   ├── 5.e.Logistic Regression-20240730T090714Z-001/
│   │   ├── 5.e.Logistic Regression/
│   │   └── Logistic Regression 2025/
│   └── KNN_Complete_Pipeline/
│
└── unsupervised_machine_learning/
    ├── Clustering/
    │   ├── K-Means_Clustering/
    │   ├── Hierarchical_Clustering_2025/
    │   ├── PCA/
    │   └── SVD/
    ├── 2025_Association_Rules/
    └── 2025_Recommendation_Engine/
```

## Contents

### Supervised Learning

| Project | Description |
|---|---|
| **Simple Linear Regression** | Predicts a single continuous target from one feature; includes a polynomial-features pipeline and a Streamlit app for live predictions. |
| **Multiple Linear Regression** | Regression on the cars dataset using multiple engineered features, with a saved preprocessing pipeline and Streamlit front end. |
| **Lasso, Ridge, Elastic Net** | Regularized regression models compared on the same cars dataset, tuned via grid search (`grid_elasticnet.pkl`). |
| **Logistic Regression** | Binary classification predicting attorney involvement in insurance claims; includes both an earlier version and a revised 2025 version with model evaluation plots (confusion matrix, ROC curve, SHAP summary) and a Streamlit app. |
| **KNN_Complete_Pipeline** | K-Nearest Neighbors classifier on the Wisconsin breast cancer dataset, including a full preprocessing + feature-selection pipeline, a Streamlit app, and Docker deployment notes. |

### Unsupervised Learning

| Project | Description |
|---|---|
| **K-Means Clustering** | Clusters universities based on their attributes, with a saved preprocessing pipeline and model. |
| **Hierarchical Clustering** | Agglomerative clustering on the same university dataset. |
| **PCA** | Dimensionality reduction on the university dataset, deployed as a Flask app. |
| **SVD** | Singular Value Decomposition for dimensionality reduction, deployed as a Flask app. |
| **Association Rules** | Market basket analysis on grocery transaction data (Apriori-style rules), deployed as a Flask app. |
| **Recommendation Engine** | Movie recommendation system built on a user-item ratings matrix, deployed as a Flask app. |

## Tech Stack

- **Language:** Python
- **ML/Data:** scikit-learn, pandas, numpy
- **Model persistence:** pickle, joblib
- **Deployment:** Streamlit, Flask
- **Database:** SQLAlchemy (MySQL) for storing prediction results in some projects

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Ayush18102002/Machine-Learning-.git
   cd Machine-Learning-
   ```
2. Navigate into the project you want to run, e.g.:
   ```bash
   cd "supervised_machine_learning/Simple Linear Regression"
   ```
3. Install the dependencies used by that project (pandas, numpy, scikit-learn, streamlit/flask, joblib, sqlalchemy, pymysql as needed):
   ```bash
   pip install pandas numpy scikit-learn streamlit flask joblib sqlalchemy pymysql
   ```
4. Run the training script, or launch the web app directly, e.g.:
   ```bash
   streamlit run streamlit_New.py
   ```
   or, for Flask-based projects:
   ```bash
   python New_PCA_flaskapp.py
   ```

> Some apps connect to a local MySQL database to log predictions — update the `user`, `pw`, and `db` values in the script to match your local setup, or remove that step if you don't need it.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.