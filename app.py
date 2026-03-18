import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="LGD Risk Predictor",
    page_icon="⚕️",
    layout="wide"
)

# 2. Title and Description
st.title("Ulcerative Colitis LGD Risk Prediction Model")
st.markdown("""
This tool uses a machine learning model (LightGBM) to predict the risk of Low-Grade Dysplasia (LGD) in patients with Ulcerative Colitis. 
Please input the patient's clinical and endoscopic features in the sidebar to generate a prediction and SHAP explanation.
""")

# 3. Load Model (Cached for performance)
@st.cache_resource
def load_model():
    # Load the model exported from R
    model = lgb.Booster(model_file='lgb_model.txt')
    return model

model = load_model()

# 4. Sidebar: Feature Input
st.sidebar.header("Patient Features")

# Continuous Variables
age_onset = st.sidebar.number_input("Age at Onset (years)", min_value=0, max_value=100, value=30, step=1)
duration = st.sidebar.number_input("Disease Duration (months)", min_value=0.0, max_value=1200.0, value=5.0, step=0.1)
esr = st.sidebar.number_input("ESR (mm/h)", min_value=0.0, max_value=1000.0, value=15.0, step=1.0)

# Categorical Variables
montreal = st.sidebar.selectbox("Montreal Classification", ["E1", "E2", "E3"])

# Updated UCEIS selectbox
uceis = st.sidebar.selectbox(
    "UCEIS Score", 
    ["Normal / Mild active (score 0-3)", "Moderate active (score 4-6)", "Severe active (score 7-8)"]
)

histology = st.sidebar.selectbox("Moderate-to-severe Histological Activity", ["No", "Yes"])

# Updated labels for Stricture and Pseudopolyp
stricture = st.sidebar.selectbox("Presence of Stricture", ["No", "Yes"])
pseudopolyp = st.sidebar.selectbox("Presence of Pseudopolyps", ["No", "Yes"])
cdiff = st.sidebar.selectbox("History of C. difficile Infection", ["No", "Yes"])

# 5. Prediction Logic
if st.sidebar.button("Predict Risk"):
    
    # Map inputs to the exact dummy variable names expected by the R-trained model
    input_dict = {
        'Ageatonset': age_onset,
        'Duration': duration,
        'ESR': esr,
        'MontrealClassificationE3': 1 if montreal == "E3" else 0,
        'UCEIS2': 1 if uceis == "Moderate active (score 4-6)" else 0,
        'UCEIS3': 1 if uceis == "Severe active (score 7-8)" else 0,
        'ModSevereHistologyYes': 1 if histology == "Yes" else 0,
        'StenosisYes': 1 if stricture == "Yes" else 0,     # Mapped to new UI variable
        'PolypYes': 1 if pseudopolyp == "Yes" else 0,      # Mapped to new UI variable
        'CdifficileYes': 1 if cdiff == "Yes" else 0
    }
    
    # Retrieve the exact feature names and order used during model training
    expected_features = model.feature_name()
    
    try:
        # Construct DataFrame strictly following the model's expected features
        input_df = pd.DataFrame([input_dict])[expected_features]
        
        # Execute prediction and UI rendering
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Prediction Result")
            with st.spinner("Calculating probability..."):
                probability = model.predict(input_df)[0]
                
                st.metric(label="Predicted Probability of LGD", value=f"{probability:.2%}")
                
                # Threshold updated to 0.186 to match the manuscript's optimal cut-off
                if probability >= 0.186:
                    st.error("⚠️ **High Risk Group**: The patient is predicted to have a high risk of LGD.")
                else:
                    st.success("✅ **Low Risk Group**: The patient is predicted to have a low risk of LGD.")
                    
                st.info("Disclaimer: This tool is for research purposes only and does not substitute professional medical advice.")

        with col2:
            st.subheader("Model Explanation (SHAP)")
            st.markdown("This waterfall plot shows how each specific feature contributed to this patient's final predicted probability.")
            
            with st.spinner("Generating SHAP plot..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                
                # Dictionary to map model's original feature names to UI display names
                name_mapping = {
                    'ModSevereHistologyYes': 'ModSevereHistology',
                    'Ageatonset': 'Ageatonset',
                    'Duration': 'Duration',
                    'ESR': 'ESR',
                    'MontrealClassificationE3': 'MontrealClassificationE3',
                    'UCEIS2': 'UCEIS2',
                    'StenosisYes': 'Stricture',
                    'PolypYes': 'Pseudopolyp',
                    'UCEIS3': 'UCEIS3',
                    'CdifficileYes': 'Cdifficile'
                }
                
                # Reassign feature names dynamically based on the model's inherent feature order
                shap_values.feature_names = [name_mapping.get(feat, feat) for feat in expected_features]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
    except KeyError as e:
        st.error(f"🚨 Feature Name Mismatch! A variable required by the model is missing or incorrectly named.")
        st.error(f"Missing variable: {e}")