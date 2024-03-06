import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import joblib
import time

# Load the trained XGBClassifier
xgb_model = joblib.load('xgb_model.joblib')
random_forest = joblib.load('randomforest1.joblib')

# Function to get user input
def get_user_input():
    st.subheader("Enter Customer Information:")
    CreditScore = st.number_input("Credit Score", min_value=0, step=1)
    Age = st.number_input("Age", min_value=0, step=1)
    Tenure = st.number_input("Tenure", min_value=0, step=1)
    Balance = st.number_input("Balance", min_value=0.0, step=1.0)
    NumOfProducts = st.number_input("NumOfProducts", min_value=0, step=1)
    HasCrCard = st.number_input("Has CrCard", min_value=0, step=1)
    IsActiveMember = st.number_input("IsActiveMember", min_value=0, step=1)
    Complain = st.number_input('Complain', min_value=0, step=1)
    Satisfaction_Score = st.number_input('Satisfaction Score', min_value=0, step=1)
    Point_Earned = st.number_input('Point Earned', min_value=0, step=1)

    features_dict = {
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'Complain': Complain,
        'Satisfaction Score': Satisfaction_Score,
        'Point Earned': Point_Earned
    }

    return pd.DataFrame([features_dict])

# Function to visualize Churn Risk Progress Bar
def churn_risk_progress_bar(churn_prob):
    st.subheader("Churn Risk Progress Bar")
    # Use a progress bar to visualize churn risk
    st.progress(float(churn_prob))  # Convert to float

    # Display churn probability as a percentage
    st.text(f"Churn Probability: {churn_prob * 100:.2f}%")

# Function for automated model questions without data selection
def automate_model_questions():
    st.subheader("Automate Model Questions")

    # Button to start automation
    if st.button("Click for Automate"):
        sample_data = pd.DataFrame({
            'CreditScore': [700, 650, 600, 720, 680],
            'Age': [35, 40, 25, 30, 45],
            'Tenure': [5, 8, 2, 7, 4],
            'Balance': [5000, 8000, 2000, 7000, 4000],
            'NumOfProducts': [2, 3, 1, 2, 1],
            'HasCrCard': [1, 1, 0, 1, 0],
            'IsActiveMember': [1, 0, 1, 1, 0],
            'Complain': [0, 1, 0, 0, 1],
            'SatisfactionScore': [4, 3, 5, 4, 2],
            'PointEarned': [20, 15, 25, 18, 12],
        })

        for _, customer in sample_data.iterrows():
            st.text(f"Credit Score: {customer['CreditScore']}")
            st.text(f"Age: {customer['Age']}")
            st.text(f"Tenure: {customer['Tenure']}")
            st.text(f"Balance: {customer['Balance']}")
            st.text(f"NumOfProducts: {customer['NumOfProducts']}")
            st.text(f"HasCrCard: {customer['HasCrCard']}")
            st.text(f"IsActiveMember: {customer['IsActiveMember']}")
            st.text(f"Complain: {customer['Complain']}")
            st.text(f"Satisfaction Score: {customer['SatisfactionScore']}")
            st.text(f"Point Earned: {customer['PointEarned']}")

            # Prepare data for prediction
            features_df = pd.DataFrame(
                {
                    'CreditScore': [customer['CreditScore']],
                    'Age': [customer['Age']],
                    'Tenure': [customer['Tenure']],
                    'Balance': [customer['Balance']],
                    'NumOfProducts': [customer['NumOfProducts']],
                    'HasCrCard': [customer['HasCrCard']],
                    'IsActiveMember': [customer['IsActiveMember']],
                    'Complain': [customer['Complain']],
                    'Satisfaction Score': [customer['SatisfactionScore']],
                    'Point Earned': [customer['PointEarned']],
                }
            )

            # Make prediction
            churn_prob = xgb_model.predict_proba(features_df)[:, 1][0]

            # Display Prediction Result
            if churn_prob > 0.5:
                st.warning("Churn Risk Detected!")
            else:
                st.success("No Churn Risk Detected.")

            # Pause for 2 seconds before moving to the next customer
            time.sleep(2)

# Streamlit App
def main():
    st.title("Customer Churn Prediction App")

    # Sidebar Navigation
    page_selection = st.sidebar.selectbox("Select Page", ["Normal Prediction", "Automated Prediction"])

    # Display selected page
    if page_selection == "Normal Prediction":
        # Model selection
        model_selection = st.sidebar.selectbox("Select Model", ["XGBoost","Random forest"])
        
        if model_selection == "XGBoost":
            model = xgb_model
        elif model_selection == "Random forest":
            model = random_forest
        else:
            st.error("Model not available. Please select a different model.")

        # Get user input
        features_df = get_user_input()

        # Check if input is valid
        if features_df is not None:
            # Make prediction
            if st.button("Predict"):
                churn_prob = model.predict_proba(features_df)[:, 1][0]

                # Display Churn Risk Progress Bar
                churn_risk_progress_bar(churn_prob)

                # Display Prediction Result
                if churn_prob > 0.5:
                    st.warning("Churn Risk Detected!")
                else:
                    st.success("No Churn Risk Detected.")

    elif page_selection == "Automated Prediction":
        automate_model_questions()

if __name__ == "__main__":
    main()

