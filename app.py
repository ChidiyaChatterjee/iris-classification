import streamlit as st
import joblib
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("iris_knn_model.pkl")

model = load_model()
species = ["setosa", "versicolor", "virginica"]

def summary_row(label, value):
    st.markdown(
        f"""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 10px;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 12px; color: #64748b;">{label}</div>
            <div style="font-size: 22px; font-weight: 700; color: #0f172a;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def result_box(title, value, color="#065f46"):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border-left: 6px solid {color};
            border-radius: 12px;
            padding: 14px 16px;
            margin-top: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            min-height: 110px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-size: 12px; color: #065f46; font-weight: 600; text-transform: uppercase;">{title}</div>
            <div style="font-size: 26px; font-weight: 700; color: #064e3b; margin-top: 8px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸", layout="wide")

st.title("Iris Flower Predictor")
st.caption("A simple machine learning dashboard for flower species classification.")

left_col, right_col = st.columns([1.5, 1.5])

with left_col:
    with st.form("iris_form"):
        st.subheader("Input Measurements")
        sepal_length = st.slider("Sepal Length", 0.0, 10.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width", 0.0, 10.0, 3.5, 0.1)
        petal_length = st.slider("Petal Length", 0.0, 10.0, 1.4, 0.1)
        petal_width = st.slider("Petal Width", 0.0, 10.0, 0.2, 0.1)
        submitted = st.form_submit_button("Predict Species", use_container_width=True)

with right_col:
    st.subheader("Flower Summary")

    summary_col, result_col = st.columns([0.5, 0.8])

    with summary_col:
        summary_row("Sepal Length", f"{sepal_length:.1f} cm")
        summary_row("Sepal Width", f"{sepal_width:.1f} cm")
        summary_row("Petal Length", f"{petal_length:.1f} cm")
        summary_row("Petal Width", f"{petal_width:.1f} cm")

    with result_col:
        if submitted:
                features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                predicted_species = species[prediction]
                confidence = max(probabilities) * 100
        
                result_left, result_right = st.columns(2)
        
                with result_left:
                    result_box("Prediction Result", predicted_species.title())
        
                with result_right:
                    result_box("Confidence", f"{confidence:.2f}%")
        
                result_df = pd.DataFrame({
                    "Species": [s.title() for s in species],
                    "Probability": probabilities * 100
                })
        
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Prediction Probabilities")
                st.bar_chart(result_df.set_index("Species")["Probability"])
        else:
                result_left, result_right = st.columns(2)
        
                with result_left:
                    result_box("Prediction Result", "Waiting")
        
                with result_right:
                    result_box("Confidence", "--")
        
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Prediction Probabilities")
                st.info("Click Predict Species to see the probability chart.")