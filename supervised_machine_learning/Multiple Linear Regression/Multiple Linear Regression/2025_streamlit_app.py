# Import necessary libraries
from PIL import Image
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import pickle, joblib
from urllib.parse import quote
import seaborn as sns
import numpy as np

# Safely load pre-trained model and preprocessing pipeline
try:
    model1 = pickle.load(open('finalmodel.pkl', 'rb'))
    clean = joblib.load('preprocessed_pipeline.pkl')
except Exception as e:
    st.error(f"Error loading model or preprocessing pipeline: {e}")
    st.stop()

def predict_MPG(data, user, pw, db):
    # Transform raw data using the saved pipeline
    transformed = clean.transform(data)
    # Create DataFrame with feature names from the pipeline
    clean1 = pd.DataFrame(transformed, columns=clean.get_feature_names_out())
    
    # Rename columns to remove prefixes (to mimic training processing)
    clean1.columns = [col.split("__")[-1] for col in clean1.columns]
    
    # Add polynomial features if not already present
    if 'HP' in clean1.columns and 'HP_squared' not in clean1.columns:
        clean1['HP_squared'] = clean1['HP'] ** 2
    if 'SP' in clean1.columns and 'SP_squared' not in clean1.columns:
        clean1['SP_squared'] = clean1['SP'] ** 2
    if 'VOL' in clean1.columns and 'VOL_squared' not in clean1.columns:
        clean1['VOL_squared'] = clean1['VOL'] ** 2

    # Drop 'WT' column if it exists (as done in training)
    if 'WT' in clean1.columns:
        clean1 = clean1.drop('WT', axis=1)

    # Reorder columns to match the training model
    # Get expected feature names from the saved model
    expected_features = model1.model.exog_names.copy()
    # Remove constant if it exists (our final model was built without it)
    if 'const' in expected_features:
        expected_features.remove('const')
    try:
        clean1 = clean1[expected_features]
    except Exception as e:
        st.error(f"Error in reordering features: {e}")
        st.stop()

    # Predict using the final model
    prediction = model1.predict(clean1)

    # Combine predictions with the original data
    final = pd.concat([pd.DataFrame(prediction, columns=['Predicted_MPG']), data.reset_index(drop=True)], axis=1)

    # Save predictions to MySQL database if credentials are provided
    try:
        engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
        final.to_sql('mpg_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)
    except Exception as e:
        st.error(f"Error saving predictions to database: {e}")

    return final

# Main Streamlit App
def main():
    # Display logo on sidebar (ensure the file is in the same folder)
    try:
        image = Image.open("AiSPRY logo.jpg")
        st.sidebar.image(image)
    except Exception as e:
        st.sidebar.warning(f"Logo not found: {e}")

    st.title("Fuel Efficiency Prediction App")
    st.sidebar.title("Upload Data and Predict")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cars Fuel Efficiency Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'])

    data = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.sidebar.warning("Please upload a CSV or Excel file.")

    html_temp2 = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add Database Credentials (Optional)</p>
    </div>
    """
    st.sidebar.markdown(html_temp2, unsafe_allow_html=True)

    user = st.sidebar.text_input("MySQL user", "Type Here")
    pw = st.sidebar.text_input("MySQL password", "Type Here", type="password")
    db = st.sidebar.text_input("MySQL database", "Type Here")

    if st.button("Predict"):
        if data is not None and not data.empty:
            with st.spinner('Predicting...'):
                result = predict_MPG(data.copy(), user, pw, db)
            st.success("Prediction Completed!")
            cm = sns.light_palette("blue", as_cmap=True)
            st.table(result.style.background_gradient(cmap=cm))
            # Download option for predictions
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predicted_mpg.csv',
                mime='text/csv',
            )
        else:
            st.warning("Please upload valid data before prediction.")

if __name__ == '__main__':
    main()
