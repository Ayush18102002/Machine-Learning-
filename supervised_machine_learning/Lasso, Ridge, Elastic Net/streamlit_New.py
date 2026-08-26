# Import libraries for data manipulation, web app creation (if applicable), and database connection (if applicable)
import pandas as pd
from PIL import Image
import streamlit as st
from sqlalchemy import create_engine
from statsmodels.tools.tools import add_constant
import pickle
import joblib
from urllib.parse import quote

# Load the trained ElasticNet model for prediction
model1 = pickle.load(open('grid_elasticnet.pkl', 'rb'))  # Load best ElasticNet model

# Load the preprocessing pipelines used during model training
clean = joblib.load('preprocessed_pipeline.pkl')  # Corrected: proper file name

# Define a function to make predictions and store them in a database
def predict_MPG(data, user, pw, db):
    # Connect to the MySQL database
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    
    # Apply preprocessing to the input data
    clean_data = pd.DataFrame(clean.transform(data), columns=clean.get_feature_names_out())
    
    # Rename columns to remove prefixes
    clean_data.columns = [col.split("__")[-1] for col in clean_data.columns]
    
    # Add a constant for intercept
    P = add_constant(clean_data)
    
    # Drop feature based on VIF analysis
    clean_data1 = P.drop('WT', axis=1)  # Corrected: WT instead of numerical__WT

    # Make predictions
    prediction = pd.DataFrame(model1.predict(clean_data1), columns=['Predict_MPG'])
    
    # Combine predictions with original input data
    final = pd.concat([prediction, data], axis=1)
    
    # Save the predictions into the database
    final.to_sql('mpg_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)
    
    return final

# Define the main function to create the Streamlit app interface
def main():
    image = Image.open("AiSPRY logo.jpg")  # Logo file
    st.sidebar.image(image)

    # Set the title for the app and sidebar
    st.title("Fuel Efficiency Prediction")
    st.sidebar.title("Fuel Efficiency Prediction")

    # HTML template for heading
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cars Fuel Efficiency Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")
    
    # File uploader
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")

    # Check if a file has been uploaded
    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)
        except:
            try:
                data = pd.read_excel(uploadedFile)
            except:
                data = pd.DataFrame()
    else:
        st.sidebar.warning("You need to upload a CSV or Excel file.")
        data = None

    # Database credentials input
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add Database Credentials</p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    
    user = st.sidebar.text_input("User", "Type Here")
    pw = st.sidebar.text_input("Password", "Type Here", type='password')
    db = st.sidebar.text_input("Database", "Type Here")
    
    result = ""

    # Predict button
    if st.button("Predict"):
        if data is not None and not data.empty:
            try:
                result = predict_MPG(data, user, pw, db)
                
                import seaborn as sns
                cm = sns.light_palette("blue", as_cmap=True)
                
                st.table(result.style.background_gradient(cmap=cm))
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("No data uploaded to predict.")

if __name__ == '__main__':
    main()
