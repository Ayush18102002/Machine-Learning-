# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import streamlit as st  # For building interactive web applications
from sqlalchemy import create_engine  # For connecting to databases
from urllib.parse import quote
import pickle, joblib  # For loading the machine learning model and preprocessing pipelines
from PIL import Image  # For displaying logo/image

# Load the machine learning model and preprocessing pipeline
model = pickle.load(open('knn.pkl', 'rb'))
preprocessing_pipeline = joblib.load('pipeline_with_feature_selection')  # This includes both preprocessing and feature selection

# Function to make predictions using the loaded model
def predict(data, user, pw, db):
    try:
        # Create engine
        engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

        # Drop ID column if it exists
        if 'id' in data.columns:
            data.drop(['id'], axis = 1, inplace = True)

        # Preprocess
        processed_data = preprocessing_pipeline.transform(data)

        # Predict
        predictions = pd.DataFrame(model.predict(processed_data), columns = ['diagnosis'])
        final = pd.concat([predictions, data.reset_index(drop = True)], axis = 1)

        # Try saving to database
        try:
            final.to_sql('cancer_predictions', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
            st.success("Predictions successfully saved to MySQL database ✅")
        except Exception as db_error:
            st.error(f"❌ Failed to save to MySQL table: {db_error}")

        return final

    except Exception as e:
        st.error(f"❌ MySQL connection or prediction failed: {e}")
        return pd.DataFrame()  # return empty DataFrame if error


# Streamlit app interface
def main():
    image = Image.open("AiSPRY logo.jpg")
    st.sidebar.image(image)

    st.title("Breast Cancer Prediction")
    st.sidebar.title("Breast Cancer Prediction")

    # Upload file section
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")

    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)
        except:
            try:
                data = pd.read_excel(uploadedFile)
            except:
                data = pd.DataFrame(uploadedFile)
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")

    # MySQL credentials
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here", type = 'password')
    db = st.sidebar.text_input("database", "Type Here")

    # Predict button
    if st.button("Predict"):
        result = predict(data, user, pw, db)

        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm))

if __name__ == '__main__':
    main()
