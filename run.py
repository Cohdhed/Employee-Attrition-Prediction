import pandas as pd
import streamlit as st
import joblib
import os

# Load model and preprocessing artifacts
script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, 'attrition_detection.joblib')

# Load the file using the absolute path
artifact = joblib.load(file_path)
cat_cols = artifact['categorical_cols']



# App title and description
st.title('Employee Attrition Prediction App')
st.write("""
Predict whether an employee is likely to leave the company based on key personal and job-related details.
This tool helps HR professionals and managers make informed, data-driven decisions to improve retention and support their workforce.
""")

# Input fields
department = st.selectbox("Department", ['', 'Sales', 'Research & Development', 'Human Resources'])
num_companies = st.number_input('Number of Companies Worked', min_value=0, step=1)
years_at_company = st.number_input('Years at Company', min_value=0, step=1)
business_travel = st.selectbox("How often do you travel for business?", ['', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
education_field = st.selectbox("Education Field", ['', 'Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
age = st.number_input('Age', min_value=0, step=1)
job_role = st.selectbox("Job Role", ['', 'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                     'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
marital_status = st.selectbox("Marital Status", ['', 'Single', 'Married', 'Divorced'])
overtime = st.selectbox("Overtime", ['', 'Yes', 'No'])


# Collect user inputs
user_input = {
    'Age': age,
    'NumCompaniesWorked': num_companies,
    'YearsAtCompany': years_at_company,
    'BusinessTravel': business_travel,
    'Department': department,
    'EducationField': education_field,
    'JobRole': job_role,
    'MaritalStatus': marital_status,
    'OverTime': overtime,
}

# Check for any missing input
def check_missing_inputs(data):
    return any(value == '' for key, value in data.items() if key in cat_cols)

# Make prediction
def make_prediction(data):
    df = pd.DataFrame([data])

    X_data = pd.DataFrame(artifact['preprocessing'].transform(df), columns=artifact['preprocessing'].get_feature_names_out())

    # Predict
    prediction = artifact['model'].predict(X_data)[0]
    return 'Yes' if prediction == 1 else 'No'

# Prediction trigger
if st.button('Predict'):
    if check_missing_inputs(user_input):
        st.error("🚫 Please fill in all the fields.")
    else:
        result = make_prediction(user_input)
        if result == 'Yes':
            st.warning("⚠️ Prediction: The employee is **likely to leave**.")
        else:
            st.success("✅ Prediction: The employee is **likely to stay**.")
