import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt



# Load trained model
model = pickle.load(open("model.pkl", "rb"))
explainer = shap.Explainer(model)


st.title("🏥 Diabetes Risk Predictor")

st.write("Enter Patient Details:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):

    input_data = np.array([[pregnancies, glucose, bp, skin,
                            insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # 🥇 Premium Risk Display
    st.subheader("📊 Risk Assessment")

    st.progress(float(probability))

    st.metric(
        label="Diabetes Risk Probability",
        value=f"{probability*100:.2f}%"
    )


    if probability < 0.3:
       st.success("🟢 LOW RISK")
       st.info("Healthy metabolic indicators. Maintain current lifestyle.")

    elif 0.3 <= probability < 0.6:
       st.warning("🟡 MODERATE RISK")
       st.info("Lifestyle modifications and monitoring recommended.")

    else:
       st.error("🔴 HIGH RISK")
       st.info("Strong indicators detected. Medical consultation advised.")



    # 🔥 Hybrid Clinical Safety Rule
    if glucose > 180 or bmi > 35:
        st.warning("⚠️ Clinical Alert: Extremely high metabolic indicators detected. Immediate medical screening recommended.")



    # ---- SHAP Explanation ----
    feature_names = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree",
    "Age"
    ]

    shap_values = explainer(input_data)

    shap_values.feature_names = feature_names


    st.subheader("🔍 Feature Impact Explanation")

    fig, ax = plt.subplots()

    # For binary classification, select class 1 (diabetes)
    shap.plots.waterfall(shap_values[0][:, 1], show=False)

    st.pyplot(fig)


    
