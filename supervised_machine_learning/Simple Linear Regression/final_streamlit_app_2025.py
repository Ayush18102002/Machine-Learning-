import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sqlalchemy import create_engine
from urllib.parse import quote
import joblib

# 1) Load the final pipeline that you saved in Simple_Linear_Regression.py
#    This pipeline handles both preprocessing and the final regression model (predicting log(AT)),
#    so we can simply pass in the raw data, and get back predicted log(AT).
final_pipe = joblib.load('final_polynomial_pipeline.pkl')

# 2) Define a function that uses the final pipeline to predict AT, writes results to DB, and returns a DataFrame
def predict_AT(data: pd.DataFrame, user: str, pw: str, db: str) -> pd.DataFrame:
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

    # final_pipe already predicts AT on its original scale
    predicted_AT = final_pipe.predict(data)

    final_df = pd.concat([
        pd.DataFrame(predicted_AT, columns=['Pred_AT']), 
        data.reset_index(drop=True)
    ], axis=1)

    final_df.to_sql('at_predictions', con=engine, if_exists='replace', chunksize=1000, index=False)

    return final_df


# 3) Define the main Streamlit app
def main():
    # Optionally display a logo in the sidebar (make sure the file path is valid)
    try:
        image = Image.open("AiSPRY logo.jpg")
        st.sidebar.image(image)
    except:
        pass

    # Main title
    st.title("AT Prediction")

    # A little styling with HTML
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">AT prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # 4) Let users upload a CSV or Excel file
    uploadedFile = st.sidebar.file_uploader(
        "Choose a file (CSV or Excel)",
        type=['csv','xlsx'],
        accept_multiple_files=False,
        key="fileUploader"
    )

    # Read the uploaded file into a DataFrame
    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)
        except:
            try:
                data = pd.read_excel(uploadedFile)
            except:
                data = pd.DataFrame()  # Fallback if neither CSV nor Excel parsing works
    else:
        st.sidebar.warning("Please upload a CSV or Excel file.")
        data = None

    # 5) Ask for DB credentials in sidebar
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add MySQL Credentials</p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)

    user = st.sidebar.text_input("User", "")
    pw = st.sidebar.text_input("Password", "", type="password")
    db = st.sidebar.text_input("Database", "")

    # 6) Predict button
    if st.button("Predict"):
        if data is None or data.empty:
            st.error("No data found. Please upload a valid CSV or Excel file.")
        elif not user or not pw or not db:
            st.error("Please provide MySQL user, password, and database name.")
        else:
            # Call our prediction function
            result_df = predict_AT(data, user, pw, db)

            # Display predictions in a visually appealing table
            import seaborn as sns
            cm = sns.light_palette("blue", as_cmap=True)
            st.table(result_df.style.background_gradient(cmap=cm))

# 7) Entry point
if __name__ == '__main__':
    main()
