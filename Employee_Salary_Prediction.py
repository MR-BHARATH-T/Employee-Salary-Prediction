# %%writefile salary_predictor_app.py

import streamlit as st
import pandas as pd
import joblib

# --- Dark Theme and Container Styling (no input element overrides) ---

st.markdown("""
<style>
body, .stApp {
    background-color: #ffffff !important;  /* light background */
    color: #000000 !important;             /* dark text */
}

/* Containers & cards styling for light theme */
.stContainer > div, .card, .prediction-result {
    background: #f9f9f9 !important;        /* very light gray container */
    border-radius: 12px;
    margin-bottom: 24px;
    padding: 13px 19px;
    box-shadow: 0 0 5px #ccccccAA;
    border: 1px solid #e0e0e0;
}

/* Uniform input, textarea, and select styles: white bg, black text */
input, textarea, select,
.stSelectbox div[role="textbox"],
.stMultiSelect div[role="textbox"] {
    background-color: #ffffff !important; /* WHITE background */
    color: #000000 !important;            /* BLACK text */
    border: 1px solid #ccc !important;
    border-radius: 6px !important;
    font-size: 1rem !important;
    padding: 10px 13px !important;
    box-sizing: border-box;
    transition: border-color 0.15s;
}

/* Focus styling for all inputs */
input:focus, textarea:focus, select:focus,
.stSelectbox div[role="textbox"]:focus,
.stMultiSelect div[role="textbox"]:focus {
    border-color: #4285f4 !important;
    outline: none !important;
}

/* Dropdown option list styling */
select option {
    background-color: #ffffff !important; /* white options background */
    color: #000000 !important;            /* black options text */
}

/* Streamlit's custom dropdown options */
.stSelectbox .css-1wa3eu0-option,
.stSelectbox [role="option"],
.stMultiSelect .css-1wa3eu0-option,
.stMultiSelect [role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Placeholder text for selects */
.stSelectbox .css-1wa3eu0-placeholder,
.stMultiSelect .css-1wa3eu0-placeholder {
    color: #666666 !important; /* dark gray placeholders */
}

/* Labels, headers */
h1, h2, h3, label, .st-bd, .stTextInput>div>label {
    color: #000000 !important;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

/* Buttons */
.stButton>button {
    font-size: 1rem;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.7rem 1.4rem;
    background: #4285f4; /* Google-blue */
    color: #fff;
    border: none;
    margin-top: 10px;
    margin-bottom: 10px;
    transition: background 0.18s;
}
.stButton>button:hover {
    background: #3367d6;
}

/* Prediction result card */
.prediction-result {
    background: #f0f0f0 !important;
    border-radius: 10px;
    color: #000000;
    padding: 22px 10px 18px 10px;
    margin: 20px auto 8px auto;
    text-align: center;
    max-width: 540px;
    box-shadow: 0 0 10px #bbbbbb88;
}
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
model = joblib.load('knn_salary_pipeline.pkl')

st.set_page_config(
    page_title="💼 Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("💼 Employee Salary Predictor")
st.markdown("<span style='font-size:20px;'>Enter employee information below for instant, professional salary prediction.</span>", unsafe_allow_html=True)

# Input Section
with st.container():
    st.subheader("Employee Details")
    col1, col2 = st.columns([1, 1])

    with col1:
        employee_id = st.text_input("🔖 Employee ID", value="", max_chars=20)
        name = st.text_input("👤 Employee Name", value="", max_chars=50)
        age = st.number_input("🎂 Age", min_value=18, max_value=70, value=30, step=1, format="%d")
        gender = st.selectbox("⚧ Gender", options=["Male", "Female"])
        department = st.selectbox("🏢 Department", options=[
            "Sales", "Engineering", "Marketing", "Finance", "HR", "Operations", "IT", "Admin"])

    with col2:
        job_title = st.text_input("💼 Job Title", value="Software Engineer", max_chars=100)
        experience_years = st.number_input("⏳ Experience (Years)", min_value=0, max_value=50, value=5, step=1, format="%d")
        education_level = st.selectbox("🎓 Education Level", options=["Bachelor", "Master", "PhD"])
        location = st.selectbox("📍 Location", options=[
            "New York", "San Francisco", "Chicago", "Austin", "Seattle", "Boston", "Atlanta"])

    with st.expander("💡 Preview Input Data", expanded=True):
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Department': department,
            'Job_Title': job_title,
            'Experience_Years': experience_years,
            'Education_Level': education_level,
            'Location': location,
            'Name': name,
            'Employee_ID': employee_id
        }])
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# Prediction Output
predict_button = st.button("🔮 Predict Salary", use_container_width=True)
if predict_button:
    try:
        pred_salary = model.predict(input_df)[0]
        st.markdown(
            f"""
            <div class="prediction-result">
                <h2>💰 Prediction Result</h2>
                <div style='font-size:19px;text-align:left;'>
                  <b>Employee Name:</b> {name if name else 'N/A'}<br>
                  <b>Employee ID:</b> {employee_id if employee_id else 'N/A'}<br>
                </div>
                <hr style='border-color:#2186FF;margin:8px auto;'>
                <p style="font-size:29px;color:#2186FF;font-weight:bold;">
                    Predicted Salary: <span style="color:#fd576c;">${pred_salary:,.2f}</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
with st.container():
    st.subheader("📂 Batch Salary Prediction")
    uploaded_file = st.file_uploader("Upload a CSV with employee details before uploading just modified the column name to Id -> Employee_ID; Name -> Name;  Age -> Age; Gender -> Gender; Department -> Department; Job_designation -> Job_Title; Experience -> Experience_Years; Educatin -> Education_Level; Location -> Lcation; If you have all the above data, then you can predict the salary for the whole dataset", type="csv", key="batch_upload")
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown("**Uploaded Data Preview:**")
            st.dataframe(batch_df.head(), use_container_width=True, hide_index=True)

            batch_preds = model.predict(batch_df)
            batch_df['Predicted_Salary'] = batch_preds

            st.markdown("**Prediction Results:**")
            st.dataframe(batch_df.head(), use_container_width=True, hide_index=True)

            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Predictions as CSV",
                data=csv,
                file_name='batch_salary_predictions.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
