import streamlit as st
import pandas as pd
import pickle
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote

# Page configuration
st.set_page_config(
    page_title="AI-Powered Attorney Prediction",
    page_icon="🤖",
    layout="wide"
)

# Sidebar - Logo and File Upload
with st.sidebar:
    # Logo at top
    st.image("C:\DS26112025\Logistic Regression\AiSPRY logo.jpg", use_container_width=True)
    st.markdown("---")
    
    # File upload section
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        help="Upload CSV or Excel file with claimant data"
    )
    
    st.markdown("---")
    
    # Database configuration section
    st.header("🗄️ Database Config")
    
    db_user = st.text_input("Username", value="root", placeholder="Enter username")
    db_name = st.text_input("Database Name", value="db1", placeholder="Enter database name")
    db_password = st.text_input("Password", type="password", placeholder="Enter password")
    
    st.markdown("---")
    
    # Prediction button
    predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

# Main content area
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
        🏢 Insurance Attorney Prediction System
    </h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>
        Predict the likelihood of claimants hiring an attorney using AI-powered analytics
    </p>
    <hr style='margin-bottom: 30px;'>
""", unsafe_allow_html=True)

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None

# Handle file upload
if uploaded_file is not None:
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.data = df
        
        # Display success message
        st.success(f"✅ File uploaded successfully! {len(df)} records loaded.")
        
        # Show sample records
        st.subheader("📊 Sample Records")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
            
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.session_state.data = None
else:
    # Welcome screen when no data is loaded
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #888;'>👈 Upload your dataset to get started</h2>
            <p style='color: #aaa; font-size: 16px;'>
                Supported formats: CSV, XLSX<br>
                Required columns: CLMSEX, CLMINSUR, SEATBELT, CLMAGE, LOSS
            </p>
        </div>
    """, unsafe_allow_html=True)

# Handle prediction
if predict_button:
    if st.session_state.data is None:
        st.error("⚠️ Please upload a dataset first!")
    else:
        try:
            with st.spinner("🔄 Running predictions..."):
                # Load the trained model and preprocessing pipeline
                model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
                pipeline = joblib.load('preprocessing_pipeline_logistic.pkl')
                
                # Get the uploaded data
                df = st.session_state.data.copy()
                
                # Expected features for prediction
                required_features = ['CLMSEX', 'CLMINSUR', 'SEATBELT', 'CLMAGE', 'LOSS']
                
                # Check if required features exist
                missing_features = [f for f in required_features if f not in df.columns]
                if missing_features:
                    st.error(f"❌ Missing required columns: {', '.join(missing_features)}")
                else:
                    # Extract features for prediction
                    X_new = df[required_features]
                    
                    # Preprocess data
                    X_preprocessed = pipeline.transform(X_new)
                    
                    # Make predictions
                    predictions = model.predict(X_preprocessed)
                    prediction_proba = model.predict_proba(X_preprocessed)[:, 1]
                    
                    # Add predictions to dataframe
                    df['ATTORNEY_PREDICTION'] = predictions
                    df['ATTORNEY_PROBABILITY'] = prediction_proba.round(4)
                    
                    # Store results
                    st.session_state.data = df
                    
                    st.success("✅ Predictions completed successfully!")
                    
                    # Display full results table
                    st.subheader("🎯 Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Save to database
                    if db_name and db_user and db_password:
                        try:
                            # Create database connection
                            pw_encoded = quote(db_password)
                            engine = create_engine(f"mysql+pymysql://{db_user}:{pw_encoded}@localhost/{db_name}")
                            
                            # Write to database
                            df.to_sql('attorney_predictions', con=engine, if_exists='replace', index=False)
                            
                            st.success(f"💾 Predictions saved to database '{db_name}' in table 'attorney_predictions'")
                        except Exception as e:
                            st.warning(f"⚠️ Could not save to database: {str(e)}")
                    else:
                        st.info("ℹ️ Database credentials not provided. Results not saved to database.")
                    
                    # Download option
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="attorney_predictions.csv",
                        mime="text/csv",
                    )
                    
        except FileNotFoundError as e:
            st.error("❌ Model files not found! Please ensure 'logistic_regression_model.pkl' and 'preprocessing_pipeline_logistic.pkl' are in the same directory.")
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        Powered by AiSPRY | Logistic Regression Model | CRISP-ML(Q) Framework
    </div>
""", unsafe_allow_html=True)
